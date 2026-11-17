import carla
from carla_env.global_path_plan import global_path_planner
import numpy as np
from scipy.interpolate import Akima1DInterpolator
import math


def smooth_interpolate_locations_passing_through(locations, num_points_per_segment=20):
    """
    对 Carla Location 点进行平滑插值，返回经过所有原始点的轨迹。

    参数:
        locations (List[carla.Location]): 原始坐标点
        num_points_per_segment (int): 每两个原始点之间插值的总点数（含两端）

    返回:
        List[carla.Location]: 平滑后的路径点（包含原始点）
    """
    if len(locations) < 2:
        return locations

    # 提取坐标数组
    points = np.array([[loc.x, loc.y, loc.z] for loc in locations])
    n = len(points)

    # 构造参数 t ∈ [0, 1]，每个原始点对应一个 t 值
    t_values = np.linspace(0, 1.5, n)

    # 创建 Akima 插值函数（每维独立插值）
    interp_x = Akima1DInterpolator(t_values, points[:, 0])
    interp_y = Akima1DInterpolator(t_values, points[:, 1])
    interp_z = Akima1DInterpolator(t_values, points[:, 2])

    # 构建采样点 t_new
    total_points = (n - 1) * num_points_per_segment + 1
    t_new = np.linspace(0, 1, total_points)

    # 插值得到新点
    interpolated = np.stack([
        interp_x(t_new),
        interp_y(t_new),
        interp_z(t_new)
    ], axis=-1)

    # 转换为 Carla Location 列表
    result = []
    for point in interpolated:
        result.append(carla.Location(
            x=float(point[0]),
            y=float(point[1]),
            z=float(point[2])
        ))

    return result


def normalize_vector(vector):
    """归一化二维向量"""
    length = math.sqrt(vector.x ** 2 + vector.y ** 2)
    if length > 0:
        return carla.Vector2D(vector.x / length, vector.y / length)
    return vector  # 如果长度为零，返回原向量


def calculate_angle_difference(current_waypoint, next_waypoint, vehicle):
    """
    计算车辆倒车朝向与路径点方向之间的角度差（弧度）

    返回:
        float: 角度差（弧度），范围[-π, π]
              正值表示车辆需要向左转才能对准路径点方向
              负值表示车辆需要向右转
    """
    vehicle_transform = vehicle.get_transform()
    vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

    # 计算车辆倒车方向向量（180°反转）
    vehicle_backward_x = -math.cos(vehicle_yaw)
    vehicle_backward_y = -math.sin(vehicle_yaw)

    # 手动归一化车辆倒车方向向量
    vehicle_backward_magnitude = math.sqrt(vehicle_backward_x ** 2 + vehicle_backward_y ** 2)
    if vehicle_backward_magnitude > 0:
        vehicle_backward_x /= vehicle_backward_magnitude
        vehicle_backward_y /= vehicle_backward_magnitude

    # 计算路径点方向向量
    waypoint_vec_x = next_waypoint.x - current_waypoint.x
    waypoint_vec_y = next_waypoint.y - current_waypoint.y

    # 手动归一化路径点方向向量
    waypoint_magnitude = math.sqrt(waypoint_vec_x ** 2 + waypoint_vec_y ** 2)
    if waypoint_magnitude > 0:
        waypoint_vec_x /= waypoint_magnitude
        waypoint_vec_y /= waypoint_magnitude

    # 计算叉积和点积（注意顺序：vehicle × waypoint）
    cross = vehicle_backward_x * waypoint_vec_y - vehicle_backward_y * waypoint_vec_x
    dot = vehicle_backward_x * waypoint_vec_x + vehicle_backward_y * waypoint_vec_y

    # 计算角度差（从车辆方向到路径点方向）
    angle = math.atan2(cross, dot)

    return angle


class WaypointsPlanner():
    def __init__(self, world, ego_vehicle, stat_pose, goal_location, scenarios):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.map = self.world.get_map()
        self.debug = self.world.debug
        self.scenarios = scenarios

        self.global_route_plan = global_path_planner(world_map=self.map, sampling_resolution=2)  # 实例化全局规划器
        self.stat_pose = stat_pose
        self.goal_location = goal_location

        if self.scenarios == "parking":
            locations = [
                carla.Location(18.914436, y=-26.694237, z=0.2),
                carla.Location(x=14.334414, y=-29.98, z=0.2),
                carla.Location(x=9.334414, y=-29.884169, z=0.2),
                carla.Location(x=6.447057, y=-29.880699, z=0.2),
                carla.Location(x=3.447057, y=-29.880699, z=0.2),
            ]
            num_points_between_each_pair = 5
            self.route_waypoints = smooth_interpolate_locations_passing_through(locations, num_points_between_each_pair)

            for waypoint in self.route_waypoints:
                self.debug.draw_point(waypoint + carla.Location(0, 0, 0.2),
                                      size=0.05, color=carla.Color(0, 255, 0), life_time=0)

            print("num of route_waypoints:", len(self.route_waypoints))

        elif self.scenarios == "ghost_static":
            control_points = [
                self.stat_pose.location,
                carla.Location(x=-90.0, y=2.8, z=0.1),
                carla.Location(x=-76.393974, y=0.0, z=0.1),
                self.goal_location,
            ]

            self.route_waypoints = []
            for i in range(len(control_points) - 1):
                segment = self.global_route_plan.search_path_way(
                    origin=control_points[i],
                    destination=control_points[i + 1]
                )
                if i > 0 and len(segment) > 0:
                    self.route_waypoints.extend(segment[1:])
                else:
                    self.route_waypoints.extend(segment)

        else:
            self.route_waypoints = self.global_route_plan.search_path_way(
                origin=self.ego_vehicle.get_location(),
                destination=self.goal_location)

            # # 显示地图路径（调试用），注意训练的时候得关闭，不然图像会有路点，有可能也能训出来，就是有点奇怪
            # for waypoint in self.route_waypoints:
            #     self.debug.draw_point(waypoint.transform.location + carla.Location(0, 0, 0.2),
            #                           size=0.05, color=carla.Color(0, 255, 0), life_time=0)

        self.current_waypoint_index = 0
        self.center_lane_deviation = 0
        self.distance_from_center = 0
        self.max_distance_from_center = 1

    def reset(self):
        self.current_waypoint_index = 0
        self.center_lane_deviation = 0
        self.distance_from_center = 0

    def get_angle_diff(self, count, scenarios):
        # Location of the car
        vehicle_location = self.ego_vehicle.get_location()
        vehicle_xy = np.array([vehicle_location.x, vehicle_location.y])
        vehicle_velocity = self.ego_vehicle.get_velocity()  # Vecor3D
        vehicle_velocity_xy = np.array([vehicle_velocity.x, vehicle_velocity.y])
        speed = np.linalg.norm(vehicle_velocity_xy)

        # Keep track of closest waypoint on the route
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],
                         self.vector(vehicle_location - wp.transform.location)[:2])
            if dot > 0.0:
                waypoint_index += 1
                # self.debug.draw_point(self.route_waypoints[waypoint_index].transform.location + carla.Location(0, 0, 2),
                #                       size=0.05, color=carla.Color(255, 0, 0), life_time=5)
            else:
                break

        self.current_waypoint_index = waypoint_index
        if self.current_waypoint_index == len(self.route_waypoints) - 1:
            print("Episode done: no more waypoints left. (frame %d)" % count)
            done, vel_s = True, 0.
            centering_factor, angle_factor = 0, 0
        else:
            # Calculate deviation from center of the lane
            self.current_waypoint = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index + 1) % len(self.route_waypoints)]
            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),
                                                              self.vector(self.next_waypoint.transform.location),
                                                              self.vector(vehicle_location))
            self.center_lane_deviation += self.distance_from_center  # 计算累加值，还没用

            # Get angle difference between closest waypoint and vehicle forward vector
            fwd = self.vector(self.ego_vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle = self.angle_diff(fwd, wp_fwd)

            # 这段代码用于计算车辆当前速度在目标路径方向上的投影速度分量，从而帮助车辆沿预定路径行驶
            # 应该也能是goal_waypoint = self.current_waypoint[0], 只是说取的点远一点更准确
            goal_waypoint = self.current_waypoint.next(5.)[0]
            # 应该也能是goal_waypoint = self.next_waypoint[0]
            next_goal_waypoint = goal_waypoint.next(0.1)  # waypoints are ever 0.02 meters
            if len(next_goal_waypoint) != 1:
                print('warning: {} waypoints (not 1)'.format(len(next_goal_waypoint)))
            if len(next_goal_waypoint) == 0:
                print("Episode done: no more waypoints left. (frame %d)" % count)
                done, vel_s = True, 0.
                centering_factor, angle_factor = 0, 0
            else:
                goal_location = goal_waypoint.transform.location
                local_goal_xy = np.array([goal_location.x, goal_location.y])
                location_ahead = next_goal_waypoint[0].transform.location
                highway_vector = np.array([location_ahead.x, location_ahead.y]) - local_goal_xy
                highway_unit_vector = np.array(highway_vector) / np.linalg.norm(highway_vector)
                vel_s = np.dot(vehicle_velocity_xy, highway_unit_vector)

                # local_goal_xy = np.array([goal_location.x, goal_location.y])
                # location_ahead = self.next_waypoint.transform.location
                # highway_vector = local_goal_xy - np.array([location_ahead.x, location_ahead.y])
                # highway_unit_vector = np.array(highway_vector) / np.linalg.norm(highway_vector)
                # vel_s = np.dot(vehicle_velocity_xy, highway_unit_vector)

                #
                # Interpolated from 1 when centered to 0 when 3 m from center
                centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
                # Interpolated from 1 when aligned with the road to 0 when +/- 40 degress of road
                angle_factor = max(1.0 - abs(self.angle / np.deg2rad(30)), 0.0)
                done = False

        # not algorithm's fault, but the simulator sometimes throws the car in the air wierdly
        if vehicle_velocity.z > 1. and count < 20:
            print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(
                vehicle_velocity.z, count))
            done = True
        if vehicle_location.z > 0.5 and count < 20:
            print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(
                vehicle_location.z, count))
            done = True
        # highway_random不用
        if scenarios != "highway_random" and self.distance_from_center > self.max_distance_from_center:
            print("Episode fail: there's too much deviation ({}), usually a simulator glitch (frame {})".format(
                self.distance_from_center, count))
            done = True

        return vel_s, speed, done, centering_factor, angle_factor
        #用来衡量沿路向的速度分量、车速、是否结束以及居中和角度对齐的程度，这个度量会用于奖励函数的计算






    def find_closest_point_parking(self, vehicle_location, trajectory_points):
        # 处理空轨迹点列表的情况
        if not trajectory_points:
            return None, None, float('inf')  # 返回None索引、None点和无穷大距离

        min_distance = float('inf')
        closest_point = None
        closest_index = -1  # 初始化最近点的索引

        # 遍历所有轨迹点，同时获取索引
        for index, point in enumerate(trajectory_points):
            # 计算当前轨迹点与自主车位置的距离
            distance = vehicle_location.distance(point)

            # 如果当前距离更小，更新最小距离、最近点和索引
            if distance < min_distance:
                min_distance = distance
                closest_point = point
                closest_index = index

        return closest_index, closest_point, min_distance

    def get_angle_diff_parking(self, count, scenarios):
        # Location of the car
        vehicle_location = self.ego_vehicle.get_location()
        vehicle_xy = np.array([vehicle_location.x, vehicle_location.y])
        vehicle_velocity = self.ego_vehicle.get_velocity()  # Vecor3D
        vehicle_velocity_xy = np.array([vehicle_velocity.x, vehicle_velocity.y])
        speed = np.linalg.norm(vehicle_velocity_xy)

        # 使用示例
        waypoint_index, closest_point, distance = self.find_closest_point_parking(vehicle_location,
                                                                                  self.route_waypoints)
        # if closest_index is not None:
        #     print(f"最近点索引: {closest_index}, 最近点: {closest_point}, 距离: {distance} 米")
        # else:
        #     print("轨迹点列表为空")

        self.current_waypoint_index = waypoint_index
        if self.current_waypoint_index == len(self.route_waypoints) - 1:
            print("Episode done: no more waypoints left. (frame %d)" % count)
            done, vel_s = True, 0.
            centering_factor, angle_factor = 0, 0
        else:
            # Calculate deviation from center of the lane
            self.current_waypoint = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index + 1) % len(self.route_waypoints)]
            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint),
                                                              self.vector(self.next_waypoint),
                                                              self.vector(vehicle_location))
            self.center_lane_deviation += self.distance_from_center  # 计算累加值，还没用

            # Get angle difference between closest waypoint and vehicle forward vector
            self.angle = calculate_angle_difference(self.current_waypoint, self.next_waypoint, self.ego_vehicle)

            # print("current_waypoint_index:", self.current_waypoint_index)
            # print("distance_from_center:", self.distance_from_center)
            # print("angle:", self.angle)

            # 停车场，不需要计算行驶距离
            vel_s = 0.

            # local_goal_xy = np.array([goal_location.x, goal_location.y])
            # location_ahead = self.next_waypoint.transform.location
            # highway_vector = local_goal_xy - np.array([location_ahead.x, location_ahead.y])
            # highway_unit_vector = np.array(highway_vector) / np.linalg.norm(highway_vector)
            # vel_s = np.dot(vehicle_velocity_xy, highway_unit_vector)

            #
            # Interpolated from 1 when centered to 0 when 3 m from center
            centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
            # Interpolated from 1 when aligned with the road to 0 when +/- 40 degress of road
            angle_factor = max(1.0 - abs(self.angle / np.deg2rad(30)), 0.0)

            done = False

        # not algorithm's fault, but the simulator sometimes throws the car in the air wierdly
        if vehicle_velocity.z > 1. and count < 20:
            print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(
                vehicle_velocity.z, count))
            done = True
        if vehicle_location.z > 0.5 and count < 20:
            print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(
                vehicle_location.z, count))
            done = True
        # highway_random不用
        if scenarios != "highway_random" and self.distance_from_center > self.max_distance_from_center:
            print(
                "Episode fail: there's too much distance deviation ({}), usually a simulator glitch (frame {})".format(
                    self.distance_from_center, count))
            done = True

        if scenarios != "highway_random" and abs(self.angle) > np.deg2rad(30):
            print("Episode fail: there's too much angle deviation ({}), usually a simulator glitch (frame {})".format(
                self.angle, count))
            done = True

        return vel_s, speed, done, centering_factor, angle_factor

    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle <= -np.pi:
            angle += 2 * np.pi
        return angle

    def distance_to_line(self, A, B, p):
        num = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom

    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])

#每帧环境会调用它拿到vel_s, speed, done, centering_factor, angle_factor用来计算奖励函数
#没有路径规划的好，只看快不快，侧着滑也是算快的