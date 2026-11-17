#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Modified for DBC paper.

import random
import glob
import os
import sys
import time
from PIL import Image
from PIL.PngImagePlugin import PngImageFile, PngInfo

import carla
import math
from PIL import Image

from dotmap import DotMap

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

from carla_env.my_agents.navigation.agent import Agent, AgentState
from carla_env.my_agents.navigation.local_planner import LocalPlanner
from carla_env.waypoints_planner import WaypointsPlanner
from carla_env.scenarios import Ghost_probe_two_static_vehicle, Ghost_probe_walk, \
    set_randome_vehicles, Parking_two_static_vehicle, Parking_walk

WEATHERS = {
    'ClearNoon': carla.WeatherParameters.ClearNoon,
    'ClearSunset': carla.WeatherParameters.ClearSunset,

    'CloudyNoon': carla.WeatherParameters.CloudyNoon,
    'CloudySunset': carla.WeatherParameters.CloudySunset,

    'WetNoon': carla.WeatherParameters.WetNoon,
    'WetSunset': carla.WeatherParameters.WetSunset,

    'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
    'MidRainSunset': carla.WeatherParameters.MidRainSunset,

    'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
    'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,

    'HardRainNoon': carla.WeatherParameters.HardRainNoon,
    'HardRainSunset': carla.WeatherParameters.HardRainSunset,

    'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
    'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
}


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

        self.start()

    def start(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

    def tick(self, timeout, retries=3):
        last_error = None
        for attempt in range(retries):
            try:
                self.frame = self.world.tick(timeout)
                data = [self._retrieve_data(q, timeout) for q in self._queues]
                assert all(x.frame == self.frame for x in data)
                return data
            except Exception as exc:
                last_error = exc
                print("CARLA sync tick failed (%d/%d): %s" % (attempt + 1, retries, exc))
                time.sleep(1.0)
        raise RuntimeError("CARLA simulator did not answer after %d sync tick attempts. "
                           "Restart the CARLA server and resume training from the latest checkpoint. "
                           "Last error: %s" % (retries, last_error))

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # 杩欓噷鐨勪綔鐢ㄥ氨鏄涪寮冪鍥涗釜閫氶亾鐨勶紝涔熷氨鏄€忔槑搴?    array = array[:, :, ::-1]  # 灏咮GR杞崲涓篟GB
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def safe_world_tick(world, timeout=60.0, label="world"):
    try:
        return world.tick(timeout)
    except RuntimeError as exc:
        print("Warning: %s tick timed out once, waiting for simulator snapshot: %s" % (label, exc))
        return world.wait_for_tick(timeout)


def actor_bbox_overlap_2d(actor_a, actor_b, margin=0.15):
    try:
        loc_a = actor_a.get_transform().transform(actor_a.bounding_box.location)
        loc_b = actor_b.get_transform().transform(actor_b.bounding_box.location)
        ext_a = actor_a.bounding_box.extent
        ext_b = actor_b.bounding_box.extent
    except Exception:
        return False

    overlap_x = abs(loc_a.x - loc_b.x) <= (ext_a.x + ext_b.x + margin)
    overlap_y = abs(loc_a.y - loc_b.y) <= (ext_a.y + ext_b.y + margin)
    return bool(overlap_x and overlap_y)


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


# 杩欎釜鍑芥暟瀹炵幇鐨勬槸鍦╬ygame绐楀彛閲岄潰鑾峰彇瀛椾綋

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


# 杩欎釜鍑芥暟鐨勭洰鐨勫氨鏄疄鐜扮敤鎴锋兂浠€涔堟椂鍊欓€€鍑哄氨浠€涔堟椂鍊欓€€鍑?

def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


# 杩欎釜鏄敤鏉ョ害鏉熷悗闈㈠ぉ姘旂浉鍏虫暟鍊肩殑杈撳叆锛屾瘮濡備簯閲忥紝闆ㄩ噺锛岄浘閲忕瓑锛岄兘闇€瑕佸湪0-100涔嬮棿

class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        # self.altitude = (70 * math.sin(self._t)) - 20  # [50, -90]
        min_alt, max_alt = [30, 90]
        self.altitude = 0.5 * (max_alt + min_alt) + 0.5 * (max_alt - min_alt) * math.cos(self._t)

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


# 杩欎釜sun灏辨槸鑷畾涔夊お闃崇殑杩愯瑙掑害锛屽洜涓篶arla鍘熺敓鐨勪笉鏄緢婊¤冻

class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 60.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, world, changing_weather_speed, weather_option="ClearNoon"):
        self.world = world
        # Allow selecting specific weather by name
        if weather_option in WEATHERS:
            self.weather = WEATHERS[weather_option]
        else:
            print(f"Warning: Weather {weather_option} not found, defaulting to ClearNoon")
            self.weather = WEATHERS['ClearNoon']
        self.reset()
        self.changing_weather_speed = changing_weather_speed
        self._sun = Sun(self.weather.sun_azimuth_angle, self.weather.sun_altitude_angle)
        self._storm = Storm(self.weather.precipitation)

    def reset(self):
        self.world.set_weather(self.weather)

    def tick(self):
        # self._sun.tick(self.changing_weather_speed)
        # self._storm.tick(self.changing_weather_speed)
        # self.weather.cloudiness = self._storm.clouds
        # self.weather.precipitation = self._storm.rain
        # self.weather.precipitation_deposits = self._storm.puddles
        # self.weather.wind_intensity = self._storm.wind
        # self.weather.fog_density = self._storm.fog
        # self.weather.wetness = self._storm.wetness
        # self.weather.sun_azimuth_angle = self._sun.azimuth
        # self.weather.sun_altitude_angle = self._sun.altitude
        self.world.set_weather(self.weather)

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)


# 濡傛灉鍙栨秷浜嗘敞閲婏紝閭ｅ氨鏄湪涓嶅悓澶╂皵瀵规ā鍨嬭繘琛岃缁?
class CarlaEnv(object):
    def __init__(self,
                 render_display=0,  # 0, 1
                 record_display_images=0,  # 0, 1
                 record_rl_images=0,  # 0, 1
                 changing_weather_speed=0.0,  # [0, +inf)
                 display_text=0,  # 0, 1
                 rl_image_size=84,
                 max_episode_steps=1000,
                 frame_skip=1,
                 fov=60,  # degrees for rl camera
                 num_cameras=1,
                 port=2000,
                 trafficManagerPort=8000,
                 town="Town05",
                 scenarios="ghost_static",
                 algorithm="sac",
                 weather="ClearNoon",
                 no_walker=False,
                 min_walkers=1,
                 max_walkers=3,
                 walker_trigger_ttc_min=1.8,
                 walker_trigger_ttc_max=3.0,
                 walker_cross_probability=1.0,
                 walker_speed_min=1.2,
                 walker_speed_max=1.8,
                 walker_direction="top_to_bottom",
                 occlusion_level="medium",
                 initial_speed=None
                 ):
        if record_display_images:
            assert render_display
        self.render_display = render_display
        self.save_display_images = record_display_images
        self.save_rl_images = record_rl_images
        # 杩欓噷杩樻湁涓€涓敼鍙樺ぉ姘旂殑閫熺巼
        self.changing_weather_speed = changing_weather_speed
        self.display_text = display_text
        self.rl_image_size = rl_image_size
        self._max_episode_steps = max_episode_steps  # DMC uses this
        self.frame_skip = frame_skip
        self.scenarios = scenarios
        self.algorithm = algorithm
        self.weather_option = weather
        self.no_walker = no_walker
        self.min_walkers = min_walkers
        self.max_walkers = max_walkers
        self.walker_trigger_ttc_min = walker_trigger_ttc_min
        self.walker_trigger_ttc_max = walker_trigger_ttc_max
        self.walker_cross_probability = walker_cross_probability
        self.walker_speed_min = walker_speed_min
        self.walker_speed_max = walker_speed_max
        self.walker_direction = walker_direction
        self.occlusion_level = occlusion_level
        print("scenarios of vehicles:{}".format(self.scenarios))
        self.num_cameras = num_cameras
        print("number of used cameras:{}".format(self.num_cameras))
        
        self.initial_speed = initial_speed # Allow external setting of initial speed

        self.actor_list = []
        # actor_list鏄敤鏉ュ瓨鍌ㄦ墍鏈夌殑actor鐨勶紝鍖呮嫭ego vehicle锛屽叾浠栬溅杈嗭紝浼犳劅鍣ㄧ瓑
        self.previous_throttle = 0
        # Previous steering value used by the smoothing penalty.
        self.previous_steer = 0
        self.previous_brake = 0
        # 涓婁竴涓猼ick鐨勬补闂ㄣ€佹柟鍚戜互鍙婅浆鍚戝€硷紝鐢ㄦ潵璁＄畻褰撳墠tick鐨勬补闂ㄣ€佹柟鍚戜互鍙婅浆鍚戝€?        # 杩欐槸涓轰簡璁＄畻褰撳墠tick鐨勬补闂ㄣ€佹柟鍚戜互鍙婅浆鍚戝€艰€岃璁＄殑

        if self.render_display:
            pygame.init()
            self.display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font = get_font()
            self.clock = pygame.time.Clock()
            # 杩欓噷鐨勮繖涓猼ick鏄敤鏉ユ帶鍒舵覆鏌撶殑棰戠巼鐨勶紝鍗虫瘡涓猼ick娓叉煋涓€娆?            # 鍗充娇鏄拰浠跨湡鍣ㄧ殑tick涓嶄竴鏍凤紝涔熶笉浼氬奖鍝嶆ā鍨嬬殑璁粌
        if self.scenarios == "ghost_static":
            self.town = "Town05"
        elif self.scenarios == "highway":
            self.town = "Town04"
        elif self.scenarios == "highway_random":
            self.town = "Town04"
        elif self.scenarios == "parking":
            self.town = "Town05"
        else:
            self.town = town
            sys.exit()
        # 杩欐槸鍦烘櫙鐨勫悕绉帮紝鐢ㄦ潵鍒ゆ柇浣跨敤鍝釜鍦板浘锛宼own4鏄珮閫熷叕璺?        # 鐒跺悗town5鏄煄甯傞亾璺?
        self.trafficManagerPort = trafficManagerPort
        self.client = carla.Client('127.0.0.1', port)
        # CARLA client timeout.
        self.client.set_timeout(200.0)
        current_world = self.client.get_world()
        current_map = current_world.get_map().name
        if current_map.endswith(self.town):
            print("Using already loaded CARLA map: {}".format(current_map))
            self.world = current_world
        else:
            print("Loading CARLA map: {} -> {}".format(current_map, self.town))
            self.world = self.client.load_world(self.town)
            safe_world_tick(self.world, timeout=120.0, label="map load")
        # 杩欐槸鍔犺浇鍒癈ARLA鏈嶅姟鍣ㄧ殑涓栫晫
        self.map = self.world.get_map()
        self.traffic_manager = self.client.get_trafficmanager(int(self.trafficManagerPort))  # 8000? which port?
        self.traffic_manager.set_synchronous_mode(True)
        # 杩欐槸灏嗕氦閫氱鐞嗗櫒璁剧疆涓哄悓姝ユā寮?        # 鍚屾妯″紡鏄寚浜ら€氱鐞嗗櫒浼氱瓑寰呮墍鏈夌殑杞﹁締閮藉埌杈句簡鐩爣浣嶇疆锛屾墠浼氱户缁墽琛屼笅涓€涓猼ick
        # 杩欐槸涓轰簡纭繚妯″瀷鐨勮缁冩槸鍩轰簬鐪熷疄鐨勪氦閫氬満鏅€岃璁＄殑
        # assert self.map.name == self.town

        # remove old vehicles and sensors (in case they survived)
        safe_world_tick(self.world, timeout=60.0, label="initial cleanup")
        actor_list = self.world.get_actors()
        # 杩欐槸鑾峰彇褰撳墠涓栫晫涓殑鎵€鏈塧ctor
        for vehicle in actor_list.filter("*vehicle*"):
            print("Warning: removing old vehicle")
            vehicle.destroy()
        for sensor in actor_list.filter("*sensor*"):
            print("Warning: removing old sensor")
            sensor.destroy()
        # 杩欐槸鍒犻櫎褰撳墠涓栫晫涓殑鎵€鏈変紶鎰熷櫒
        # 浼犳劅鍣ㄥ寘鎷浉鏈猴紝婵€鍏夐浄杈剧瓑
        # 杩欐槸涓轰簡纭繚妯″瀷鐨勮缁冩槸鍩轰簬鐪熷疄鐨勪氦閫氬満鏅€岃璁＄殑锛岃繖閲屼负浠€涔?        # 杩欓噷涓轰粈涔堣鍏堣繘琛屽垹闄ゆ搷浣滐紝鏄洜涓鸿瀵瑰簲鍓嶉潰鐨勫厛鑾峰彇鍐嶅垹闄ゅ悧锛?        # ego
        if self.scenarios == "highway":
            self.stat_pose = carla.Transform(carla.Location(x=7.746319, y=-176.374084, z=0.500000),
                                             carla.Rotation(pitch=0.000000, yaw=-90.224854, roll=0.000000))
            self.goal_location = carla.Location(x=160.372131, y=-371.640350, z=0.200000)

        elif self.scenarios == "highway_random":
            self.stat_pose = carla.Transform(carla.Location(x=-479.593018, y=340.415558, z=0.200000),
                                             carla.Rotation(pitch=0.000000, yaw=57.992210, roll=0.000000))

            self.goal_location = carla.Location(x=8.823779, y=174.374573, z=0.006000)

        elif self.scenarios == "ghost_static":
            self.stat_pose = carla.Transform(carla.Location(x=-111.12045288085938, y=2.8197994232177734, z=0.1))
            self.goal_location = carla.Location(x=-76.393974, y=-25.0, z=0.1)

        elif self.scenarios == "parking":
            self.stat_pose = carla.Transform(carla.Location(x=18.914436, y=-26.694237, z=0.2),
                                             carla.Rotation(pitch=-0.010655, yaw=41.727516, roll=0.015250))
            self.goal_location = carla.Location(x=6.447057, y=-29.880699, z=0.2)




        else:
            sys.exit()
        self.vehicle = None
        self.vehicle_start_pose = None
        self.vehicles_list = []  # their ids
        # 杩欐槸涓€涓垪琛紝鐢ㄦ潵瀛樺偍鎵€鏈夌殑杞﹁締id
        self.vehicles = None
        self.reset_vehicle()  # creates self.vehicle
        self.actor_list.append(self.vehicle)
        self.waypoints_planner = WaypointsPlanner(self.world, self.vehicle, self.stat_pose, self.goal_location,
                                                  self.scenarios)
        # 杩欓噷闈㈠嚭鐜颁簡WaypointsPlanner杩欎釜绫伙紝杩欐槸涓€涓粈涔堟牱鐨勭被
        self.image_front = None
        self.image_back = None

        blueprint_library = self.world.get_blueprint_library()
        self.last_dist_to_goal = 1e10

        if render_display:
            bp = blueprint_library.find('sensor.camera.rgb')
            bp.set_attribute('enable_postprocess_effects', str(True))
            self.camera_rgb = self.world.spawn_actor(bp, carla.Transform(carla.Location(x=-5.5, z=2.8),
                                                                         carla.Rotation(pitch=-15)),
                                                     attach_to=self.vehicle)
            self.actor_list.append(self.camera_rgb)

        # we'll use up to five cameras, which we'll stitch together
        bp = blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.rl_image_size))
        bp.set_attribute('image_size_y', str(self.rl_image_size))
        bp.set_attribute('fov', str(fov))
        bp.set_attribute('enable_postprocess_effects', str(True))
        location = carla.Location(x=1.6, y=0, z=1.7)
        self.camera_rl = self.world.spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=0.0)),
                                                attach_to=self.vehicle)
        
        # Add TCP camera: 900x256 front
        bp_tcp = blueprint_library.find('sensor.camera.rgb')
        bp_tcp.set_attribute('image_size_x', str(900))
        bp_tcp.set_attribute('image_size_y', str(256))
        bp_tcp.set_attribute('fov', str(100))
        self.camera_tcp = self.world.spawn_actor(
            bp_tcp,
            carla.Transform(
                carla.Location(x=1.3, y=0.0, z=2.3),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            ),
            attach_to=self.vehicle
        )
        self.actor_list.append(self.camera_tcp)
        self.image_tcp = None
        self.camera_tcp.listen(lambda image: self._on_tcp_image(image))

        self.camera_rl_left = self.world.spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=-float(fov))),
                                                     attach_to=self.vehicle)
        self.camera_rl_lefter = self.world.spawn_actor(bp,
                                                       carla.Transform(location, carla.Rotation(yaw=-2 * float(fov))),
                                                       attach_to=self.vehicle)
        self.camera_rl_right = self.world.spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=float(fov))),
                                                      attach_to=self.vehicle)
        self.camera_rl_righter = self.world.spawn_actor(bp,
                                                        carla.Transform(location, carla.Rotation(yaw=2 * float(fov))),
                                                        attach_to=self.vehicle)

        self.camera_rl_back = self.world.spawn_actor(bp, carla.Transform(carla.Location(x=-1.6, y=0, z=1.7),
                                                                         carla.Rotation(yaw=180)),
                                                     attach_to=self.vehicle)
        self.actor_list.append(self.camera_rl)
        self.actor_list.append(self.camera_rl_left)
        self.actor_list.append(self.camera_rl_lefter)
        self.actor_list.append(self.camera_rl_right)
        self.actor_list.append(self.camera_rl_righter)
        self.actor_list.append(self.camera_rl_back)
        # 杩欎釜鍚庤鐨勬憚鍍忓ご鏄簲鐢ㄥ湪鍊掕溅涔嬩腑鐨?        # 杩欓噷闈㈠嚭鐜颁簡5涓憚鍍忓ご锛屽垎鍒槸rgb銆乺l銆乺l_left銆乺l_lefter銆乺l_right銆乺l_righter銆乺l_back

        # 杩欐槸璁剧疆浜?涓憚鍍忓ご锛岃瀵熷緱鏇村姞鍏ㄩ潰
        # 鐒跺悗璁剧疆浜嗕竴涓鎾炰紶鎰熷櫒锛屾娴嬭嚜杞﹀拰鍒殑鐜鏈夋病鏈夊彂鐢熺鎾烇紝杩樻湁灏辨槸涓€涓帇绾夸紶鎰熷櫒锛?        # 杩樿缃簡鍘嬬嚎鍜岀鎾炵瓑绾э紝杩欎釜绛夌骇鍙互鐢ㄦ潵璁＄畻鍚庨潰鐨勫鍔卞嚱鏁?
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        # Attach collision callback.
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        # 璁剧疆鐩戝惉鍑芥暟锛岄亣鍒扮鎾炲氨浼氳皟鐢╛on_collision鍑芥暟
        self.actor_list.append(self.collision_sensor)
        self._collision_intensities_during_last_time_step = []
        # 杩欐槸涓€涓垪琛紝鐢ㄦ潵瀛樺偍鑷溅鍜屽埆鐨勭幆澧冨彂鐢熺鎾炵殑寮哄害锛岃繖涓繛寮哄害涔熷彲浠ユ祴閲忓憖

        bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        # 杩欐槸涓€涓溅閬?invasion 浼犳劅鍣紝鐢ㄦ潵妫€娴嬭嚜杞︽槸鍚﹁繚鍙嶄簡杞﹂亾瑙勫垯
        self.lane_invasion_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(lambda event: self._on_invasion(event))
        # 鐩戝惉鍑芥暟锛岀洃鍚槸鍚︽湁鍘嬬嚎
        self.actor_list.append(self.lane_invasion_sensor)
        self._lane_invasion_intensities_during_last_time_step = []
        # 杩欐槸涓€涓垪琛紝鐢ㄦ潵瀛樺偍鑷溅杩濆弽杞﹂亾瑙勫垯鐨勫己搴︼紝杩欎釜涔熷彲浠ユ祴閲忓憖

        if self.save_display_images or self.save_rl_images:
            import datetime
            now = datetime.datetime.now()
            image_dir = "images-" + now.strftime("%Y-%m-%d-%H-%M-%S")
            os.mkdir(image_dir)
            self.image_dir = image_dir

        if self.render_display:
            self.sync_mode = CarlaSyncMode(self.world, self.camera_rgb, self.camera_rl, self.camera_rl_left,
                                           self.camera_rl_lefter, self.camera_rl_right, self.camera_rl_righter,
                                           self.camera_rl_back, fps=20)
        else:
            self.sync_mode = CarlaSyncMode(self.world, self.camera_rl, self.camera_rl_left, self.camera_rl_lefter,
                                           self.camera_rl_right, self.camera_rl_righter, self.camera_rl_back, fps=20)

        # weather
        self.weather = Weather(self.world, self.changing_weather_speed, self.weather_option)

        # dummy variables given bisim's assumption on deep-mind-control suite APIs
        low = -1.0
        high = 1.0
        self.action_space = DotMap()
        # Define action-space bounds for the RL API.
        self.action_space.low.min = lambda: low
        self.action_space.high.max = lambda: high
        self.action_space.shape = [2]
        # 瀹氫箟RL鏅鸿兘浣撶殑鍔ㄤ綔绌洪棿锛岃繖閲岀殑鍔ㄤ綔绌洪棿鏄竴涓?缁村悜閲忥紝鍒嗗埆鏄补闂ㄥ拰鍒硅溅
        self.observation_space = DotMap()
        self.observation_space.shape = (3, rl_image_size, num_cameras * rl_image_size)
        self.observation_space.dtype = np.dtype(np.uint8)
        self.reward_range = None
        self.metadata = None
        self.action_space.sample = lambda: np.random.uniform(low=low, high=high,
                                                             size=self.action_space.shape[0]).astype(np.float32)
        # 鍦ㄨ繖閲屽氨鏄殢鏈洪噰鏍峰姩浣滅┖闂翠腑鐨勪竴涓姩浣滐紝鐢ㄦ潵鎺㈢储
        # roaming carla agent
        self.agent = None
        # 鍏堝垱寤哄嚭鏉ワ紝鍚庨潰鍐嶈繘琛屽疄渚嬪寲
        self.count = 0
        # 杩欐槸涓€涓鏁板櫒锛岀敤鏉ヨ褰昍L鏅鸿兘浣撶殑姝ユ暟
        self.dist_s = 0
        # 杩欐槸涓€涓窛绂讳紶鎰熷櫒锛岀敤鏉ヨ褰曡嚜杞﹀拰鐩爣鐐圭殑璺濈
        self.return_ = 0
        self.last_yaw = self.vehicle.get_transform().rotation.yaw
        # 杩欐槸涓€涓鍔卞嚱鏁帮紝鐢ㄦ潵璁板綍RL鏅鸿兘浣撶殑濂栧姳
        self.collide_count = 0
        self.walker_collision = False
        self.lane_invasion_count = 0
        if self.scenarios == "ghost_static":
            self.collide_count_max = 5
            # 杩欐槸涓€涓鎾炰紶鎰熷櫒锛岀敤鏉ヨ褰曡嚜杞﹀拰鍒殑鐜鍙戠敓纰版挒鐨勬鏁帮紝杩欎釜娆℃暟鍙互鐢ㄦ潵璁＄畻濂栧姳鍑芥暟
            self.lane_invasion_count_max = 50
        elif self.scenarios == "highway":
            self.collide_count_max = 20
            self.lane_invasion_count_max = 5
        elif self.scenarios == "highway_random":
            self.collide_count_max = 20
            self.lane_invasion_count_max = 5
        elif self.scenarios == "parking":
            self.collide_count_max = 5
            self.lane_invasion_count_max = 5
        else:
            sys.exit()
        # 杩欐槸涓€涓洖鍚堥噷闈㈠満鏅殑鏈€澶ч檺鍒剁鎾炴鏁板拰韪╃嚎娆℃暟锛屽鏋滆秴杩囦簡杩欎釜灏辫繘鍏ヤ笅涓€涓洖鍚?
        self.velocities = []
        # 杩欐槸涓€涓垪琛紝鐢ㄦ潵瀛樺偍鑷溅鐨勯€熷害
        self.world.tick()
        self.reset()  # creates self.agent
        # 涓婇潰瀹氫箟浜哛L鏅鸿兘浣撶殑鍔ㄤ綔绌洪棿鍜岃瀵熺┖闂达紝杩欓噷鐨勫姩浣滅┖闂存槸涓€涓?缁村悜閲忥紝鍒嗗埆鏄补闂ㄥ拰鍒硅溅

    def _on_tcp_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.image_tcp = array[:, :, ::-1] # BGR to RGB

    def get_tcp_inputs(self):
        # 1. RGB
        if self.image_tcp is None:
            rgb = np.zeros((256, 900, 3), dtype=np.uint8)
        else:
            rgb = self.image_tcp

        # 2. Speed
        v = self.vehicle.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2 + v.z**2) # m/s

        if self.scenarios == "ghost_static":
            # The ghost-probe route is a straight road; provide TCP with a
            # stable forward target instead of relying on the generic planner.
            target_point = [0.0, 20.0]
            command = 4
            return rgb, speed, target_point, command

        # 3. Target Point and Command
        try:
            # Note: next_wp and next_cmd logic from TCPAgent
            # We assume WaypointsPlanner or Agent can provide this
            next_wp, next_cmd = self.waypoints_planner.run_step() 
            ego_trans = self.vehicle.get_transform()
            ego_loc = ego_trans.location
            ego_rot = ego_trans.rotation
            theta = np.radians(ego_rot.yaw + 90)
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
                ])
            local_tp = np.array([next_wp[0]-ego_loc.x, next_wp[1]-ego_loc.y])
            local_tp = R.T.dot(local_tp)
            target_point = [float(local_tp[0]), float(local_tp[1])]
            command = int(next_cmd.value) if hasattr(next_cmd, 'value') else 4
        except:
            target_point = [0.0, 0.0]
            command = 4

        return rgb, speed, target_point, command

    def _on_collision(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)

        # 妫€鏌ユ槸鍚︽挒鍒颁簡琛屼汉
        is_walker = "walker" in event.other_actor.type_id
        if is_walker:
            print("CRITICAL: Collision with WALKER!")
            self.walker_collision = True

        # print('Collision (intensity {})'.format(intensity))
        if self.scenarios == "ghost_static":
            if is_walker:
                # 鎾炶浜虹粰浜堟瀬澶х殑鎯╃綒鍊?(2000)锛屾姌鍚?Reward -20
                self._collision_intensities_during_last_time_step.append(2000)
            else:
                self._collision_intensities_during_last_time_step.append(1)
        elif self.scenarios == "highway":
            self._collision_intensities_during_last_time_step.append(intensity)
        elif self.scenarios == "highway_random":
            self._collision_intensities_during_last_time_step.append(intensity)
        elif self.scenarios == "parking":
            self._collision_intensities_during_last_time_step.append(1)
        else:
            sys.exit()

    def _on_invasion(self, event):
        center_weight = 10  # 榛勮壊涓嚎锛氫竴娆＄洿鎺ヨ婊★紝绔嬪嵆缁撴潫涓€鍥炲悎
        side_weight = 1     # 鍏朵粬瀹炵嚎锛氫繚鎸佸師鏉ョ殑鏉冮噸

        for marking in event.crossed_lane_markings:
            type_str = str(marking.type)

            if "Solid" not in type_str and "Vegetation" not in type_str:
                continue

            # 灏濊瘯璇嗗埆榛勮壊涓嚎锛欳ARLA 涓粍鑹茶溅閬撶嚎涓€鑸负 Yellow 棰滆壊
            mark_color = getattr(marking, "color", None)
            if mark_color == carla.LaneMarkingColor.Yellow:
                self._lane_invasion_intensities_during_last_time_step.append(center_weight)
            else:
                self._lane_invasion_intensities_during_last_time_step.append(side_weight)

    def reset(self):
        self.waypoints_planner.reset()
        # 杩欐槸涓轰簡閲嶇疆璺緞瑙勫垝鍣紝鎶婅矾寰勮鍒掑櫒鐨勭姸鎬侀噸缃负鍒濆鐘舵€?        # waypoints_planner鏄竴涓矾寰勮鍒掑櫒锛岀敤鏉ヨ鍒掕嚜杞︾殑璺緞
        self.reset_vehicle()
        # Tick once after resetting the ego vehicle.
        self.world.tick()
        # 杩欐槸涓轰簡绛夊緟鍦烘櫙鍔犺浇瀹屾垚锛岀‘淇濊嚜杞﹀拰鍏朵粬鐜鐨勭姸鎬佹槸涓€鑷寸殑
        # world.tick()鐨勪綔鐢ㄦ槸鏇存柊鍦烘櫙鐨勭姸鎬侊紝纭繚鑷溅鍜屽叾浠栫幆澧冪殑鐘舵€佹槸涓€鑷寸殑
        if self.scenarios == "highway_random":
            self.reset_random_vehicles()
        elif self.scenarios == "ghost_static":
            self.walker_probe = None
            self.walker_probes = []
            self.walker_states = {}
            self.walker_flag = False
            self.walker_decision_made = False
            self.walker_collision = False
            self.reset_ghost_scenarios()
        elif self.scenarios == "parking":
            self.reset_parking_scenarios()
            pass

        self.world.tick()
        self.agent = RoamingAgentModified(self.vehicle, follow_traffic_lights=False)
        # Episode step counter.
        self.count = 0
        # Travel-distance accumulator.
        self.dist_s = 0
        # Episode return accumulator.
        self.return_ = 0
        # Per-episode velocity history.
        self.velocities = []
        # 娓呯┖涓婁釜鍥炲悎鐨勯€熷害璁板綍
        self.last_dist_to_goal = 1e10
        # 杩欐槸涓€涓窛绂讳紶鎰熷櫒锛岀敤鏉ヨ褰曡嚜杞﹀拰鐩爣鐐圭殑璺濈锛岃繖鏄灏戠背
        obs, _, _, _ = self.step(action=None)
        return obs

    def reset_vehicle(self):
        # I modified this function
        self.vehicle_start_pose = self.stat_pose
        if self.scenarios == "ghost_static":
            conflict_x = -76.393974
            conflict_y = float(self.stat_pose.location.y)
            start_x = conflict_x - random.uniform(25.0, 40.0)
            candidate = carla.Location(x=float(start_x), y=conflict_y, z=float(self.stat_pose.location.z))
            try:
                wp = self.map.get_waypoint(candidate, project_to_road=True, lane_type=carla.LaneType.Driving)
            except Exception:
                wp = self.map.get_waypoint(candidate, project_to_road=True)
            if wp is not None:
                self.vehicle_start_pose = wp.transform
                self.vehicle_start_pose.location.z = float(self.stat_pose.location.z)
        if self.vehicle is None:
            # create vehicle
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprint = blueprint_library.find('vehicle.tesla.model3')
            self.vehicle = self.world.spawn_actor(vehicle_blueprint, self.vehicle_start_pose)
        else:
            self.vehicle.set_transform(self.vehicle_start_pose)

        # 鏍规嵁澶╂皵鑷姩鎺у埗杞︾伅锛氬湪鍌嶆櫄/涓嬮洦/婀胯矾绛変綆鑳借搴﹀満鏅笅鎵撳紑杩戝厜鐏?绀哄鐏?        light_enum = None
        if hasattr(carla, "VehicleLightState"):
            light_enum = carla.VehicleLightState
        elif hasattr(carla, "libcarla") and hasattr(carla.libcarla, "VehicleLightState"):
            light_enum = carla.libcarla.VehicleLightState

        if light_enum is not None:
            need_lights = False
            if (self.weather_option.endswith("Sunset") or
                    "Rain" in self.weather_option or
                    "Wet" in self.weather_option):
                need_lights = True

            if need_lights:
                lights = light_enum.Position | light_enum.LowBeam
                self.vehicle.set_light_state(light_enum(lights))
            else:
                # 鐧藉ぉ鏅存湕绛夊満鏅紝鍏抽棴杞︾伅
                self.vehicle.set_light_state(light_enum(0))

        # carla > 0.9.10 :
        # 璁剧疆鍒濋€熷害 (渚嬪 20 km/h 鈮?5.55 m/s)
        if self.initial_speed is not None:
            initial_speed = self.initial_speed
        else:
            if self.scenarios == "ghost_static":
                initial_speed = 6.0
            else:
                initial_speed = 5.55  # m/s
        
        yaw = self.vehicle_start_pose.rotation.yaw
        # 灏嗚搴﹁浆鎹负寮у害
        yaw_rad = math.radians(yaw)
        # 璁＄畻閫熷害鍚戦噺
        velocity_vector = carla.Vector3D(
            x=initial_speed * math.cos(yaw_rad),
            y=initial_speed * math.sin(yaw_rad),
            z=0.0
        )
        self.vehicle.set_target_velocity(velocity_vector)
        self.vehicle.set_target_angular_velocity(carla.Vector3D())
        # carla <= 0.9.8  :
        # self.vehicle.set_velocity(carla.Vector3D())
        # self.vehicle.set_angular_velocity(carla.Vector3D())

        # 杩欐槸閲嶆柊瀹氫箟涓€涓氦閫氬伐鍏?
    def reset_random_vehicles(self):
        # clear out old vehicles
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        # Tick once after resetting the ego vehicle.
        self.world.tick()
        # Clear cached traffic vehicle ids after destroying them.
        self.vehicles_list = []
        # 杩欓噷涔熷寘鍚嚜杞﹀悧锛?
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)

        road_id = 45
        # Number of background vehicles for the random highway scenario.
        num_vehicles = 5
        # 杩欐槸瀹氫箟浜嗕竴涓溅杈嗘暟閲忥紝鐢ㄦ潵琛ㄧず鍦烘櫙涓竴鍏辨湁澶氬皯杈嗚溅
        other_car_waypoints = []
        # 杩欐槸瀛樺偍鍏朵粬杞﹁締鐨剋aypoint锛寃aypoint鏄竴涓偣锛岀敤鏉ヨ〃绀鸿溅杈嗙殑浣嶇疆
        for _ in range(num_vehicles):
            lane_id = random.choice([-1, -2, -3, -4])
            vehicle_s = np.random.uniform(100., 200.)
            other_car_waypoints.append(self.map.get_waypoint_xodr(road_id, lane_id, vehicle_s))
        # 杩欐槸涓烘瘡杈嗚溅璁惧畾涓€涓獁aypoint锛寃aypoint鏄竴涓偣锛岀敤鏉ヨ〃绀鸿溅杈嗙殑浣嶇疆锛?        # 鍦ㄥ悗闈㈢殑鐨勪唬鐮侀噷闈㈢敓鎴愯溅杈嗘椂浼氱敤鍒拌繖涓唬鐮?
        # Spawn vehicles
        # carla>0.9.10
        # for n, waypoint in enumerate(other_car_waypoints):
        #     transform = waypoint.transform
        #     transform.location.z += 0.2
        #     blueprint_library = self.world.get_blueprint_library()
        #     other_vehicle = set_randome_vehicles(blueprint_library, self.world, transform)
        #     if other_vehicle is not None:
        #         if random.uniform(-0.5, 1) > 0:
        #             # other_vehicle.set_autopilot(True, self.trafficManagerPort)  # carla>0.9.10
        #         else:
        #             # other_vehicle.set_autopilot(False, self.trafficManagerPort)  # carla>0.9.10
        #
        #         self.vehicles_list.append(other_vehicle)
        # 杩欐娉ㄩ噴鐨勪唬鐮佺殑浣滅敤鏄粈涔?        # 0.9.8
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        batch = []
        for n, waypoint in enumerate(other_car_waypoints):
            transform = waypoint.transform
            transform.location.z += 0.1
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                pass
                # logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        self.traffic_manager.global_percentage_speed_difference(30.0)

    def reset_ghost_scenarios(self):
        # clear out old vehicles
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        self.world.tick()
        self.vehicles_list = []

        blueprint_library = self.world.get_blueprint_library()
        try:
            other_vehicle_1, other_vehicle_2, other_vehicle_3 = Ghost_probe_two_static_vehicle(blueprint_library,
                                                                                               self.world,
                                                                                               self.occlusion_level)
            self.vehicles_list.append(other_vehicle_1)
            self.vehicles_list.append(other_vehicle_2)
            self.vehicles_list.append(other_vehicle_3)
            
            self.walker_probes = []
            self.walker_states = {}
            if not self.no_walker:
                min_walkers = max(0, int(self.min_walkers))
                max_walkers = max(min_walkers, int(self.max_walkers))
                num_walkers = random.randint(min_walkers, max_walkers)
                ped_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")

                conflict_x = -76.393974
                y_top = 7.885394
                y_bottom = -2.885394

                ego_start = self.vehicle_start_pose.location
                ego_x = float(ego_start.x)
                ego_y = float(ego_start.y)

                occluders = [other_vehicle_1, other_vehicle_2, other_vehicle_3]
                occluders = [v for v in occluders if v is not None]

                occluder_candidates = []
                for v in occluders:
                    try:
                        v_loc = v.get_location()
                    except Exception:
                        continue
                    if ego_x < float(v_loc.x) < (conflict_x - 0.5):
                        occluder_candidates.append(v)

                for _ in range(num_walkers):
                    if self.walker_direction == "random":
                        from_top = random.random() < 0.5
                    else:
                        from_top = self.walker_direction != "bottom_to_top"
                    yaw = -90.0 if from_top else 90.0

                    occ = random.choice(occluder_candidates) if occluder_candidates else (random.choice(occluders) if occluders else None)
                    if occ is not None:
                        try:
                            occ_loc = occ.get_location()
                            occ_x = float(occ_loc.x)
                            occ_y = float(occ_loc.y)
                        except Exception:
                            occ = None

                    spawn_x = float(conflict_x + random.uniform(-0.3, 0.3))
                    spawn_y = float((y_top if from_top else y_bottom) + random.uniform(-0.6, 0.6))

                    if occ is not None and from_top:
                        denom = (occ_x - ego_x)
                        if abs(denom) > 1e-3:
                            aligned_y = ego_y + (occ_y - ego_y) * ((spawn_x - ego_x) / denom)
                            spawn_y = float(np.clip(aligned_y + random.uniform(-0.25, 0.25), 6.0, 8.8))
                            spawn_y = float(max(spawn_y, occ_y + 0.9))

                    ped = self.world.try_spawn_actor(
                        random.choice(ped_blueprints),
                        carla.Transform(
                            carla.Location(x=float(spawn_x), y=float(spawn_y), z=0.2),
                            carla.Rotation(yaw=float(yaw))
                        )
                    )
                    if ped is None:
                        continue

                    self.walker_probes.append(ped)
                    self.walker_states[ped.id] = {
                        "flag": False,
                        "decision_made": False,
                        "will_cross": (random.random() < self.walker_cross_probability),
                        "trigger_ttc": random.uniform(self.walker_trigger_ttc_min, self.walker_trigger_ttc_max),
                        "crossing_yaw": yaw,
                        "target_location": [float(conflict_x), float(y_bottom if from_top else y_top)]
                    }

                    control = carla.WalkerControl()
                    control.speed = 0.0
                    control.direction = carla.Vector3D(0.0, 0.0, 0.0)
                    ped.apply_control(control)

                    self.vehicles_list.append(ped)

            self.walker_probe = self.walker_probes[0] if self.walker_probes else None
        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.vehicles_list])
        # 閲嶆柊瀹氫箟涓嬩竴娆￠鎺㈠ご鐨勮缁冨満鏅紝鍖呮嫭杞﹁締鍜岄亾璺?
    def reset_parking_scenarios(self):
        # clear out old vehicles
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        self.world.tick()
        self.vehicles_list = []

        blueprint_library = self.world.get_blueprint_library()
        try:
            other_vehicle_1, other_vehicle_2 = Parking_two_static_vehicle(blueprint_library, self.world)
            self.vehicles_list.append(other_vehicle_1)
            self.vehicles_list.append(other_vehicle_2)
            self.walker_probe = Parking_walk(blueprint_library, self.world)
            self.vehicles_list.append(self.walker_probe)

        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.vehicles_list])
        # 閲嶆柊鍒濆鍖栦笅涓€娆＄殑鍋滆溅鍦烘櫙璁粌锛屽寘鎷溅杈嗗拰閬撹矾

    # test env
    def compute_steer_action(self):
        control = self.agent.run_step()  # PID decides control.steer
        # 杩欎釜杈撳嚭鐨勬槸carla鐨勮嚜甯︾殑鑷畾涔夊璞★紝鍖呮嫭閫熷害锛岃搴︼紝娌归棬锛屽埞杞︾瓑锛岃繖涓嚱鏁扮殑鐩殑灏辨槸灏?        # 娌归棬鍜屽埞杞﹀悎骞舵垚涓€涓彉閲忥紝鐒跺悗瀹氫箟涓恒€?1锛?銆?        # control鍙橀噺鎺ユ敹浜嗕笁涓暟鍊硷紝鍒嗗埆鏄痵teer銆乼hrottle銆乥rake
        steer = control.steer
        # 杞悜
        throttle = control.throttle
        # 娌归棬
        brake = control.brake
        throttle_brake = -brake
        if throttle > 0.:
            throttle_brake = throttle
        # 鐢ㄤ竴涓垽鏂鍙ユ潵瀹炵幇锛屽鏋滄补闂ㄥぇ浜?锛岄偅涔堟补闂╛brake灏辩瓑浜庢补闂紝鍚﹀垯灏辩瓑浜庡埞杞︾殑璐熷€?        steer_action = np.array([steer, throttle_brake], dtype=np.float32)
        return steer_action

    # 杩欐浠ｇ爜鐨勪綔鐢ㄥ氨鏄皢carla鐨勮嚜甯︾殑鑷畾涔夊璞¤浆鎹㈡垚numpy鏁扮粍锛屾柟渚垮悗缁殑璁＄畻銆傚洜涓篟L
    # 浠栧彧璁ゆ暟缁勪笉璁ゅ埆鐨?
    def step(self, action):
        rewards = []
        for _ in range(self.frame_skip):  # default 1
            next_obs, reward, done, info = self.simulator_step(action)
            rewards.append(reward)

            # RL鐨勫喅绛栨槸姣旀瘡涓€甯х殑棰戠巼瑕佸皯寰楀锛岀劧鍚庡鏋滄瘡涓€甯ч兘杩涜鍐崇瓥鐨勮瘽锛?            # 灏变細闄峰叆璁＄畻鍐椾綑鐨勫洶闅句箣涓幓
            # 杩欓噷鐨勮繖浜涗釜甯ч兘浣跨敤鍚屼竴涓姩浣滐紝姣忎釜甯ц繑鍥炲悇鑷殑濂栧姳

            if done:
                break
        return next_obs, np.mean(rewards), done, info  # just last info?

    def simulator_step(self, action, dt=0.05):
        info = {
            "success": False,
            "collision": False,
            "dist_to_walker": -1.0,
            "speed": 0.0,
            "yaw_diff": 0.0
        }
        if self.scenarios == "ghost_static":
            #
            vehicle_velocity = self.vehicle.get_velocity()  # Vecor3D
            vehicle_velocity_xy = np.array([vehicle_velocity.x, vehicle_velocity.y])
            ego_speed = np.linalg.norm(vehicle_velocity_xy)
            # 杩欓噷鏄绠楄溅杈嗗湪娌℃湁棰犵案鐨勬儏鍐典笅鐨勯€熷害锛屾墍浠ヤ粬杩欓噷娌℃湁鐢▃鍧愭爣鐨勯€熷害锛宔go_speed鏄钩鏂瑰樊鐨勯€熷害

            vehicle_location = np.array([self.vehicle.get_location().x, self.vehicle.get_location().y], dtype=np.float32)

            vehicle_transform = self.vehicle.get_transform()
            fwd_vector = vehicle_transform.get_forward_vector()

            conflict_point = np.array([-76.393974, float(self.stat_pose.location.y)], dtype=np.float32)
            vec_to_conflict = conflict_point - vehicle_location
            long_dist_conflict = fwd_vector.x * vec_to_conflict[0] + fwd_vector.y * vec_to_conflict[1]
            key_segment = (-5.0 < long_dist_conflict < 20.0)

            walkers = getattr(self, "walker_probes", None)
            if not walkers:
                walkers = [self.walker_probe] if self.walker_probe is not None else []

            closest_walker = None
            closest_dist = float("inf")

            for walker in list(walkers):
                if walker is None:
                    continue
                try:
                    w_loc = walker.get_location()
                except Exception:
                    continue

                current_position = np.array([w_loc.x, w_loc.y], dtype=np.float32)
                dist_from_vehicle_location = np.linalg.norm(vehicle_location - current_position)

                if dist_from_vehicle_location < closest_dist:
                    closest_dist = dist_from_vehicle_location
                    closest_walker = walker

                if not hasattr(self, "walker_states"):
                    self.walker_states = {}
                state = self.walker_states.get(walker.id)
                if state is None:
                    state = {"flag": False, "decision_made": False}
                    self.walker_states[walker.id] = state

                will_cross = state.get("will_cross", True)
                trigger_ttc = state.get("trigger_ttc")
                if trigger_ttc is None:
                    trigger_ttc = random.uniform(self.walker_trigger_ttc_min, self.walker_trigger_ttc_max)
                    state["trigger_ttc"] = trigger_ttc

                target = state.get("target_location")
                if target is None:
                    target = [-76.393974, -2.885394]
                target_np = np.array(target, dtype=np.float32)
                dist_from_target = np.linalg.norm(target_np - current_position)

                ego_speed_safe = max(float(ego_speed), 0.1)
                ttc_to_conflict = long_dist_conflict / ego_speed_safe
                if (not state["flag"]) and (not state["decision_made"]) and will_cross and key_segment and (0.0 < ttc_to_conflict < trigger_ttc):
                    state["decision_made"] = True

                    speed_min = max(0.2, float(self.walker_speed_min))
                    speed_max = max(speed_min, float(self.walker_speed_max))
                    speed = random.uniform(speed_min, speed_max)

                    pedestrain_control = carla.WalkerControl()
                    pedestrain_control.speed = float(speed)
                    yaw = float(state.get("crossing_yaw", -90.0))
                    pedestrain_rotation = carla.Rotation(0, yaw, 0)
                    pedestrain_control.direction = pedestrain_rotation.get_forward_vector()
                    walker.apply_control(pedestrain_control)
                    state["flag"] = True

                if state["flag"] and (dist_from_target < 1.0):
                    control = carla.WalkerControl()
                    control.direction.x = 0
                    control.direction.z = 0
                    control.direction.y = 0
                    walker.apply_control(control)
                    state["flag"] = False

            self.walker_probe = closest_walker

        elif self.scenarios == "parking":
            random_speed = random.random()
            pedestrain_control = carla.WalkerControl()
            # pedestrain_control.speed = random_speed + 0.5
            pedestrain_control.speed = random_speed + 3.0
            pedestrain_rotation = carla.Rotation(0, -90, 0)
            pedestrain_control.direction = pedestrain_rotation.get_forward_vector()
            self.walker_probe.apply_control(pedestrain_control)
            pass
        # 璁剧疆琛屼汉鍦╬arking杩欎釜鍦板浘涓婃槸鎬庝箞琛岃蛋鐨?        # control鏉ヨ幏鍙栧弬鏁帮紝apply_control鏉ュ簲鐢ㄥ弬鏁?
        if self.render_display:
            if should_quit():
                return
            self.clock.tick()

        if action is not None:
            steer = float(action[0])

            steer = steer * 0.45

            if abs(steer) < 0.15:
                steer = 0.0

            throttle_brake = float(action[1])

            # steer = 0.01
            # throttle_brake = 0.3

            if throttle_brake >= 0.0:
                throttle = throttle_brake
                brake = 0.0
            else:
                throttle = 0.0
                brake = -throttle_brake

            assert 0.0 <= throttle <= 1.0
            assert -1.0 <= steer <= 1.0
            assert 0.0 <= brake <= 1.0
            if self.scenarios == "parking":
                vehicle_control = carla.VehicleControl(
                    throttle=throttle,
                    steer=steer,
                    brake=brake,
                    hand_brake=False,
                    reverse=True,
                    manual_gear_shift=False
                )
            # 杞﹁締鎺у埗鍣紝鍜屼笂闈㈤偅涓槸涓€鏍风殑

            else:
                if self.algorithm == "change":  # no used
                    # ddqn, ppo锛屽叾浠栫畻娉曚篃鍙互璇曡瘯
                    vehicle_control = carla.VehicleControl(
                        throttle=self.previous_throttle * 0.9 + throttle * 0.1,
                        steer=self.previous_steer * 0.9 + steer * 0.1,
                        brake=self.previous_brake * 0.9 + brake * 0.1,
                        hand_brake=False,
                        reverse=False,
                        manual_gear_shift=False
                    )
                else:
                    # Smooth steering to suppress left-right oscillation.
                    steer = self.previous_steer * 0.9 + steer * 0.1

                    vehicle_control = carla.VehicleControl(
                        throttle=throttle,
                        steer=steer,
                        brake=brake,
                        hand_brake=False,
                        reverse=False,
                        manual_gear_shift=False
                    )

            self.vehicle.apply_control(vehicle_control)
        else:
            throttle, steer, brake = 0., 0., 0.

        self.previous_throttle = throttle
        self.previous_steer = steer
        self.previous_brake = brake

        # Advance the simulation and wait for the data.
        if self.render_display:
            snapshot, image_rgb, image_rl, image_rl_left, image_rl_lefter, image_rl_right, image_rl_righter, \
            image_rl_back = self.sync_mode.tick(timeout=20.0, retries=3)
        else:
            snapshot, image_rl, image_rl_left, image_rl_lefter, image_rl_right, image_rl_righter, image_rl_back = self.sync_mode.tick(
                timeout=20.0, retries=3)

        info = {}
        info['reason_each_episode_ended'] = ''  # 杩欎釜鍙互鑷繁鍔狅紝灏辨槸娓告垙done鐨勫師鍥?        # 1.Center line deviation penalty
        if self.scenarios == "parking":
            vel_s, speed, done, centering_factor, angle_factor = self.waypoints_planner.get_angle_diff_parking(
                self.count,
                self.scenarios)
        else:
            vel_s, speed, done, centering_factor, angle_factor = self.waypoints_planner.get_angle_diff(self.count,
                                                                                                       self.scenarios)
        # 2.Collision penalty
        collision_intensities_during_last_time_step = sum(self._collision_intensities_during_last_time_step)
        self._collision_intensities_during_last_time_step.clear()  # clear it ready for next time step
        assert collision_intensities_during_last_time_step >= 0.
        colliding = float(collision_intensities_during_last_time_step > 0.)
        if colliding:
            self.collide_count += 1
        else:
            self.collide_count = 0
        if self.collide_count >= self.collide_count_max:
            print("Episode fail: too many collisions ({})! (frame {})".format(speed, self.collide_count))
            done = True
            info['collision'] = True
            info['reason_each_episode_ended'] = 'collision'
        # 3.Line pressure punishment
        lane_invasion_times_during_last_time_step = sum(self._lane_invasion_intensities_during_last_time_step)
        self._lane_invasion_intensities_during_last_time_step.clear()  # clear it ready for next time step
        assert lane_invasion_times_during_last_time_step >= 0.
        self.lane_invasion_count = lane_invasion_times_during_last_time_step
        if self.scenarios != "highway_random" and self.lane_invasion_count >= self.lane_invasion_count_max:
            print("Episode fail: too many lane_invasion ({})! (frame {})".format(speed, self.lane_invasion_count))
            done = True
            info['reason_each_episode_ended'] = 'lane_invasion'

        vehicle_location = self.vehicle.get_location()
        vehicle_xy = np.array([vehicle_location.x, vehicle_location.y])
        goal_xy = np.array([self.goal_location.x, self.goal_location.y])
        dist_to_goal = np.linalg.norm(vehicle_xy - goal_xy)
        pedestrian_collision_penalty = 0.0
        if self.scenarios == "ghost_static":
            walkers = getattr(self, "walker_probes", None)
            if not walkers:
                walkers = [self.walker_probe] if self.walker_probe is not None else []

            min_walker_distance = float("inf")
            pedestrian_bbox_overlap = False
            for walker in list(walkers):
                if walker is None:
                    continue
                try:
                    w_loc = walker.get_location()
                except RuntimeError:
                    continue
                d = math.sqrt((w_loc.x - vehicle_location.x) ** 2 +
                              (w_loc.y - vehicle_location.y) ** 2 +
                              (w_loc.z - vehicle_location.z) ** 2)
                min_walker_distance = min(min_walker_distance, d)
                pedestrian_bbox_overlap = pedestrian_bbox_overlap or actor_bbox_overlap_2d(self.vehicle, walker)

            if min_walker_distance < float("inf"):
                info["dist_to_walker"] = min_walker_distance

            if self.walker_collision or pedestrian_bbox_overlap:
                print("Episode fail: pedestrian collision ({:.2f} m). (frame {})".format(
                    min_walker_distance, self.count))
                done = True
                info["collision"] = True
                info["pedestrian_collision"] = True
                info["pedestrian_bbox_overlap"] = pedestrian_bbox_overlap
                info["reason_each_episode_ended"] = "pedestrian_collision"
                pedestrian_collision_penalty = 200.0
        success_bonus = 0.0
        if self.scenarios == "ghost_static" and not done:
            conflict_x = -76.393974
            passed_conflict_distance = float(vehicle_location.x) - conflict_x
            if passed_conflict_distance > 1.0:
                print("Episode success: safely passed ghost-probe conflict area. (frame %d)" % self.count)
                done, vel_s = True, 0.
                info['success'] = True
                info['reason_each_episode_ended'] = 'passed_conflict_area'
                success_bonus = 30.0
        # print("dist_to_goal:", dist_to_goal)
        if dist_to_goal < 0.5:
            print("Episode done: arrive the goal. (frame %d)" % self.count)
            done, vel_s = True, 0.
            info['success'] = True
            info['reason_each_episode_ended'] = 'goal'

        # Closer to the goal
        if dist_to_goal < self.last_dist_to_goal:
            # Optimized reward function: R = min(1.0 / max(d, d_min), R_max)
            d_min = 0.1
            R_max = 10.0
            reward_dist_to_goal = min(1.0 / max(dist_to_goal, d_min), R_max)
        else:
            reward_dist_to_goal = -0.001 * dist_to_goal

        # 4. Traffic Light Stop Line Termination
        # 妫€鏌ユ槸鍚﹀埌杈剧孩缁跨伅鍓嶇殑鐧界嚎 (鍋滄绾?
        if self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            if traffic_light and traffic_light.get_state() == carla.TrafficLightState.Red:
                stop_waypoints = traffic_light.get_stop_waypoints() if hasattr(traffic_light, "get_stop_waypoints") else []
                if len(stop_waypoints) > 0:
                    # 鎵惧埌璺濈鑷溅鏈€杩戠殑鍋滄绾跨偣
                    min_dist_to_stop_line = float('inf')
                    for wp in stop_waypoints:
                        d = vehicle_location.distance(wp.transform.location)
                        if d < min_dist_to_stop_line:
                            min_dist_to_stop_line = d

                    # 濡傛灉璺濈灏忎簬闃堝€?(渚嬪 2.0 绫? 涓旈€熷害杈冧綆 (鍙€?锛屽垯缁撴潫鍥炲悎
                    # Treat reaching the red-light stop line as success.
                    if min_dist_to_stop_line < 2.0:
                        print("Episode success: reached traffic light stop line. (frame %d)" % self.count)
                        done = True
                        info['success'] = True
                        # Extra reward for stopping at the stop line.
                        reward_dist_to_goal += 10.0
        # print("dist_to_goal:", dist_to_goal)
        # print("reward_dist_to_goal:", reward_dist_to_goal)

        self.last_dist_to_goal = dist_to_goal
        # 锛侊紒锛侊紒濡傛灉涓嶈€冭檻瀵艰埅绾匡紝waypoints_planner.py鍋忕瀵艰埅绾胯€岀粓姝㈢殑鏉′欢瑕佹敞閲婃帀銆?reward鍋氱浉搴旂殑淇敼
        if self.scenarios == "ghost_static":
            collision_cost = 0.01 * collision_intensities_during_last_time_step
            lane_invasion_cost = 0.005 * lane_invasion_times_during_last_time_step
            # 淇锛歝ollision_cost 搴旇鏄儵缃氶」锛堝噺鍙凤級锛岃€屼笉鏄鍔遍」锛堝姞鍙凤級
            # 璋冩暣锛?            # 1. 鍔犲叆璺濈濂栧姳 reward_dist_to_goal锛岀粰浜堟鍚戞縺鍔?            # 2. 鍘绘帀杞悜鎯╃綒锛屽洜涓哄凡缁忓湪鐗╃悊灞傚仛浜嗗己骞虫粦锛屽噺灏戣礋鍙嶉
            # 3. 鍔犲叆 centering_factor 鍜?angle_factor锛屽鍔辫溅杈嗕繚鎸佺洿绾胯椹讹紝鎶戝埗鎽嗗姩

            # Penalize unnecessary crawling, but do not punish braking for a pedestrian hazard.
            low_speed_penalty = 0.0
            pedestrian_in_danger_zone = False

            # 妫€娴嬭浜烘槸鍚﹀湪杞﹀ご鍓嶆柟锛屽鏋滃湪鍓嶆柟涓斿埞杞︼紝缁欎簣濂栧姳
            braking_reward = 0.0
            speed_holding_reward = 0.0
            safety_distance_reward = 0.0
            # 纭繚 brake 鍙橀噺瀛樺湪 (闃叉 reset 鏃?action 涓?None 瀵艰嚧鎶ラ敊)
            current_brake = brake if 'brake' in locals() else 0.0

            if self.walker_probe:
                v_loc = self.vehicle.get_location()
                w_loc = self.walker_probe.get_location()

                # 璁＄畻杞﹁締鍓嶆柟鍚戦噺
                fwd_vector = self.vehicle.get_transform().get_forward_vector()
                vec_to_walker = carla.Vector3D(w_loc.x - v_loc.x, w_loc.y - v_loc.y, w_loc.z - v_loc.z)
                dist_to_walker = math.sqrt(vec_to_walker.x ** 2 + vec_to_walker.y ** 2 + vec_to_walker.z ** 2)
                info['dist_to_walker'] = dist_to_walker

                long_dist = fwd_vector.x * vec_to_walker.x + fwd_vector.y * vec_to_walker.y + fwd_vector.z * vec_to_walker.z

                # 妯悜璺濈
                lat_dist = math.sqrt(max(0, dist_to_walker ** 2 - long_dist ** 2))

                # 鍒ゅ畾鏉′欢锛?                # 1. 绾靛悜璺濈鍦ㄥ墠鏂?0-15 绫冲唴
                # 2. 妯悜璺濈鍦ㄥ乏鍙?2.0 绫冲唴 (绾︿负鍗婃潯杞﹂亾瀹?+ 瀹夊叏浣欓噺)
                if (0 < long_dist < 15.0) and (lat_dist < 1.5):
                    pedestrian_in_danger_zone = True
                    # 鍗遍櫓鍖哄煙浼樺寲锛?                    # 1. 缁х画鍚戝墠鐨勬儵缃?(Throttle): e^(x^2)
                    # 鍋囪 x 涓烘补闂ㄥ€?(0-1)
                    danger_scale = (15.0 - long_dist) / 15.0
                    brake_target = float(np.clip(danger_scale, 0.2, 1.0))
                    forward_penalty = 2.0 * throttle * danger_scale

                    # 2. 鍒逛綇杞︾殑濂栧姳 (Brake): x^2
                    # 鍋囪 x 涓哄埞杞﹀€?(0-1)
                    stop_reward = 2.0 * current_brake * brake_target

                    # 缁煎悎濂栧姳
                    braking_reward = stop_reward - forward_penalty
                    if dist_to_walker > 5.0:
                        safety_distance_reward = 1.0
                    elif dist_to_walker > 3.0:
                        safety_distance_reward = 0.2
                    elif dist_to_walker > 1.5:
                        safety_distance_reward = -2.0 * (3.0 - dist_to_walker) / 1.5
                    else:
                        safety_distance_reward = -5.0
                else:
                    # 琛屼汉涓嶅湪鍓嶆柟鍗遍櫓鍖哄煙
                    # 1. 鏄惧紡娓呴浂鍒硅溅濂栧姳
                    braking_reward = 0.0

                    # 2. 棰濆鎯╃綒鏃犳剰涔夌殑鍒硅溅锛氭病浜嬪埆韪╁埞杞︼紒
                    # 濡傛灉鍦ㄥ畨鍏ㄥ尯鍩熻俯鍒硅溅锛岀粰浜堟儵缃?(brake 鑼冨洿 0-1)
                    if current_brake > 0.0:
                        braking_reward -= 0.5 * current_brake

                    # 3. 瀹夊叏鍖哄煙锛氬己鍒朵繚鎸?30km/h (绾?8.33 m/s) 鍖€閫熺洿琛?                    # 浣跨敤姝ｅ悜濂栧姳锛氫娇鐢ㄩ珮鏂嚱鏁板舰寮忥紝閫熷害瓒婃帴杩戠洰鏍囧€煎鍔辫秺楂橈紝宄板€?2.0
                    target_speed = 6.0
                    speed_holding_reward = 1.0 * math.exp(-0.5 * (speed - target_speed) ** 2)

            if speed < 1.0 and not pedestrian_in_danger_zone:
                low_speed_penalty = 0.5

            # 璁＄畻 yaw 鍙樺寲 (杞﹀ご鏅冨姩鎯╃綒)
            current_yaw = self.vehicle.get_transform().rotation.yaw
            # 澶勭悊瑙掑害璺冲彉 (渚嬪 179 -> -179)
            yaw_diff = current_yaw - self.last_yaw
            if yaw_diff > 180: yaw_diff -= 360
            if yaw_diff < -180: yaw_diff += 360
            info['yaw_diff'] = abs(yaw_diff)
            yaw_penalty = 0.5 * abs(yaw_diff)  # 绯绘暟鍙皟锛屾牴鎹?step 棰戠巼
            self.last_yaw = current_yaw

            steer_delta = steer - self.previous_steer
            steer_smoothness_penalty = 1.0 * abs(steer_delta)

            reward = vel_s * dt + reward_dist_to_goal + success_bonus + centering_factor + angle_factor - collision_cost - lane_invasion_cost - pedestrian_collision_penalty - low_speed_penalty + braking_reward + speed_holding_reward + safety_distance_reward - 0.5 * abs(
                steer) ** 2 - yaw_penalty - steer_smoothness_penalty
        elif self.scenarios == "highway":
            collision_cost = 0.001 * collision_intensities_during_last_time_step
            # reward = vel_s * dt + reward_dist_to_goal - collision_cost - 0.1 * brake - 0.1 * abs(steer)
            if self.algorithm == "ddqn":
                reward = vel_s * dt + centering_factor + angle_factor + reward_dist_to_goal
            elif self.algorithm == "ppo":
                reward = vel_s * dt + centering_factor + angle_factor + reward_dist_to_goal
            # sac,ddpg,td3锛屽鑸换鍔? 0.1 * abs(steer)鍙互鍘讳簡锛屽洜涓鸿矾绾垮彲鑳芥槸寰堝集
            elif self.algorithm == "sac":
                reward = vel_s * dt + reward_dist_to_goal + centering_factor - collision_cost - 0.1 * abs(steer)
            elif self.algorithm == "ddpg":
                reward = vel_s * dt + reward_dist_to_goal + centering_factor - collision_cost - 0.1 * abs(steer)
            elif self.algorithm == "td3":
                reward = vel_s * dt + reward_dist_to_goal + centering_factor - collision_cost - 0.1 * abs(steer)
            else:
                print("unsupported reward algorithm")
                sys.exit()
        elif self.scenarios == "highway_random":
            collision_cost = 0.001 * collision_intensities_during_last_time_step
            if self.algorithm == "sac":
                reward = vel_s * dt - collision_cost - 0.1 * abs(steer)
            else:
                print("highway_random only tested for sac绠楁硶, you can try others by yourself!")
                sys.exit()
        elif self.scenarios == "parking":
            collision_cost = 0.001 * collision_intensities_during_last_time_step
            if self.algorithm == "ddqn" or self.algorithm == "sac":
                speed_diff = math.fabs(2.0 - speed)
                reward = centering_factor + angle_factor + reward_dist_to_goal - speed_diff - collision_cost - 0.1 * abs(
                    steer)
            else:
                print("highway_random only tested for ddqn or sac绠楁硶, you can try others by yourself!")
                sys.exit()
        else:
            print("unsupported scenario")
            sys.exit()

        self.dist_s += vel_s * dt
        self.return_ += reward
        # print("dist_s:", self.dist_s)

        self.weather.tick()

        # Draw the display.
        if self.render_display:
            draw_image(self.display, image_rgb)
            if self.display_text:
                self.display.blit(self.font.render('frame %d' % self.count, True, (255, 255, 255)), (8, 10))
                self.display.blit(self.font.render(
                    'highway progression %4.1f m/s (%5.2f m) (%5.2f speed)' % (vel_s, self.dist_s, speed), True,
                    (255, 255, 255)), (8, 28))
                self.display.blit(self.font.render('%5.2f meters off goal' % dist_to_goal, True, (255, 255, 255)),
                                  (8, 46))
                self.display.blit(
                    self.font.render('%5.2f reward (return %.1f)' % (reward, self.return_), True, (255, 255, 255)),
                    (8, 64))
                self.display.blit(
                    self.font.render('%5.2f collision intensity ' % collision_intensities_during_last_time_step, True,
                                     (255, 255, 255)), (8, 82))
                self.display.blit(
                    self.font.render('%5.2f thottle, %5.2f steer, %5.2f brake' % (throttle, steer, brake), True,
                                     (255, 255, 255)), (8, 100))
                self.display.blit(self.font.render(str(self.weather), True, (255, 255, 255)), (8, 118))
            pygame.display.flip()

        # get pixel image obs
        rgbs = []
        self.image_front = image_rl
        self.image_back = image_rl_back

        if self.num_cameras == 1:
            if self.scenarios == "parking":
                ims = [image_rl_back]
            else:
                ims = [image_rl]
        elif self.num_cameras == 3:
            ims = [image_rl_left, image_rl, image_rl_right]
        elif self.num_cameras == 5:
            ims = [image_rl_lefter, image_rl_left, image_rl, image_rl_right, image_rl_righter]
        else:
            raise ValueError("num cameras must be 1 or 3 or 5")
        for im in ims:
            bgra = np.array(im.raw_data).reshape(self.rl_image_size, self.rl_image_size, 4)  # BGRA format
            bgr = bgra[:, :, :3]  # BGR format (84 x 84 x 3)
            rgb = np.flip(bgr, axis=2)  # RGB format (84 x 84 x 3)
            rgbs.append(rgb)
        rgb = np.concatenate(rgbs, axis=1)  # (84 x 252 x 3)
        # rgb: (84, 84, 3)
        # print("rgb:", rgb.shape)

        # Rowan added
        if self.render_display and self.save_display_images:
            image_name = os.path.join(self.image_dir, "display%08d.jpg" % self.count)
            pygame.image.save(self.display, image_name)
            # ffmpeg -r 20 -pattern_type glob -i 'display*.jpg' carla.mp4
        if self.save_rl_images:
            image_name = os.path.join(self.image_dir, "rl%08d.png" % self.count)
            im = Image.fromarray(rgb)
            metadata = PngInfo()
            metadata.add_text("throttle", str(throttle))
            metadata.add_text("steer", str(steer))
            metadata.add_text("brake", str(brake))
            im.save(image_name, "PNG", pnginfo=metadata)

            # # Example usage:
            # from PIL.PngImagePlugin import PngImageFile
            # im = PngImageFile("rl00001234.png")
            # # Actions are stored in the image's metadata:
            # print("Actions: %s" % im.text)
            # throttle = float(im.text['throttle'])  # range [0, 1]
            # steer = float(im.text['steer'])  # range [-1, 1]
            # brake = float(im.text['brake'])  # range [0, 1]
        self.count += 1

        next_obs = rgb  # (84 x 84 x 3) or (84 x 252 x 3) or (84 x 420 x 3)
        # debugging - to inspect images:
        # import matplotlib.pyplot as plt
        # import pdb; pdb.set_trace()
        # plt.imshow(next_obs)
        # plt.show()
        next_obs = np.transpose(next_obs, [2, 0, 1])  # 3 x 84 x 84/252/420
        assert next_obs.shape == self.observation_space.shape
        if self.count >= self._max_episode_steps:
            print("Episode success: I've reached the episode horizon ({}).".format(self._max_episode_steps))
            done = True
            info['reason_each_episode_ended'] = 'horizon'
        if speed < 0.02 and self.count >= 100 and self.count % 100 == 0:
            print("Episode fail: speed too small ({}), think I'm stuck! (frame {})".format(speed, self.count))
            done = True
            info['reason_each_episode_ended'] = 'stuck'

        info['crash_intensity'] = collision_intensities_during_last_time_step
        info['throttle'] = throttle
        info['steer'] = steer
        info['brake'] = brake
        info['distance'] = vel_s * dt
        info['speed'] = speed

        # Add acceleration info
        acceleration = self.vehicle.get_acceleration()
        info['acceleration'] = math.sqrt(acceleration.x ** 2 + acceleration.y ** 2 + acceleration.z ** 2)

        return next_obs, reward, done, info

    def finish(self):
        print('destroying actors.')
        for actor in self.actor_list:
            actor.destroy()
        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        time.sleep(0.5)
        pygame.quit()
        print('done.')

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        if self.scenarios == "parking":
            bgra = np.array(self.image_back.raw_data).reshape(self.rl_image_size, self.rl_image_size, 4)  # BGRA format
        else:
            bgra = np.array(self.image_front.raw_data).reshape(self.rl_image_size, self.rl_image_size, 4)  # BGRA format
        bgr = bgra[:, :, :3]  # BGR format (84 x 84 x 3)
        rgb = np.flip(bgr, axis=2)  # RGB format (84 x 84 x 3)

        rgb_array = rgb.astype(np.uint8)  # 杞崲涓?uint8 绫诲瀷

        # 灏?numpy 鏁扮粍杞崲涓?PIL 鍥惧儚
        image = Image.fromarray(rgb_array, 'RGB')

        # 浣跨敤 Pillow 鐨?resize 鏂规硶杩涜璋冩暣澶у皬
        resized_image = image.resize((256, 256), Image.BILINEAR)

        # 灏?PIL 鍥惧儚杞崲鍥?numpy 鏁扮粍
        resized_array = np.array(resized_image)

        return resized_array


class LocalPlannerModified(LocalPlanner):

    def __del__(self):
        pass  # otherwise it deletes our vehicle object

    def run_step(self):
        return super().run_step(debug=False)  # otherwise by default shows waypoints, that interfere with our camera


class RoamingAgentModified(Agent):
    """
    RoamingAgent implements a basic agent that navigates scenes making random
    choices when facing an intersection.

    This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, follow_traffic_lights=True):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(RoamingAgentModified, self).__init__(vehicle)
        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        self._follow_traffic_lights = follow_traffic_lights

        # for throttle 0.5, 0.75, 1.0
        args_lateral_dict = {
            'K_P': 1.0,
            'K_D': 0.005,
            'K_I': 0.0,
            'dt': 1.0 / 20.0}
        opt_dict = {'lateral_control_dict': args_lateral_dict}

        self._local_planner = LocalPlannerModified(self._vehicle, opt_dict)

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state and self._follow_traffic_lights:
            if debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        if hazard_detected:
            control = self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            control = self._local_planner.run_step()

        return control


if __name__ == '__main__':
    env = CarlaEnv(
        render_display=1,  # 0, 1
        record_display_images=1,  # 0, 1
        record_rl_images=1,  # 0, 1
        changing_weather_speed=1.0,  # [0, +inf)
        display_text=0,  # 0, 1
        frame_skip=4,
        max_episode_steps=100000,
        rl_image_size=84
    )

    try:
        done = False
        while not done:
            action = env.compute_steer_action()
            next_obs, reward, done, info = env.step(action)
        obs = env.reset()

    finally:
        env.finish()
