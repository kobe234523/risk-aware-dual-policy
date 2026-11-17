import carla
import random
import numpy as np


def Ghost_probe_two_static_vehicle(blueprint_library, world, occlusion_level="medium"):
    if occlusion_level == "weak":
        vehicle_xs = [-92.393974, -84.983971, -70.393974]
    elif occlusion_level == "strong":
        vehicle_xs = [-88.393974, -79.483971, -76.893974]
    else:
        vehicle_xs = [-89.393974, -80.983971, -72.393974]

    #
    spawn_point = carla.Transform(carla.Location(x=vehicle_xs[0], y=6.247448, z=0.100000),
                                  carla.Rotation(pitch=0.000000, yaw=-0.139465, roll=0.000000))
    bp_vehicle = random.choice(blueprint_library.filter('vehicle.audi.tt'))
    other_vehicle_1 = world.try_spawn_actor(bp_vehicle, spawn_point)

    spawn_point = carla.Transform(carla.Location(x=vehicle_xs[1], y=6.285394, z=0.100000),
                                  carla.Rotation(pitch=0.000000, yaw=-0.139465, roll=0.000000))
    bp_vehicle = random.choice(blueprint_library.filter('vehicle.carlamotors.carlacola'))
    other_vehicle_2 = world.try_spawn_actor(bp_vehicle, spawn_point)

    spawn_point = carla.Transform(carla.Location(x=vehicle_xs[2], y=6.247448, z=0.100000),
                                  carla.Rotation(pitch=0.000000, yaw=-0.139465, roll=0.000000))
    bp_vehicle = random.choice(blueprint_library.filter('vehicle.carlamotors.carlacola'))
    other_vehicle_3 = world.try_spawn_actor(bp_vehicle, spawn_point)
    

    return other_vehicle_1, other_vehicle_2, other_vehicle_3


def Ghost_probe_walk(blueprint_library, world):
    pedestrain_blueprints = world.get_blueprint_library().filter("walker.pedestrian.0001")
    pedestrain = world.try_spawn_actor(random.choice(pedestrain_blueprints),
                                       carla.Transform(carla.Location(x=-76.393974, y=7.885394, z=0.2),
                                                       carla.Rotation(yaw=-96.712929)))

    return pedestrain


def set_randome_vehicles(blueprint_library, world, transform):
    bp_vehicle = random.choice(blueprint_library.filter('vehicle.*'))
    other_vehicle = world.try_spawn_actor(bp_vehicle, transform)

    return other_vehicle


# Transform(Location(x=18.914436, y=-26.694237, z=0.143540), Rotation(pitch=-0.010655, yaw=41.727516, roll=0.015250))

#  Transform(Location(x=9.334414, y=-29.884169, z=0.200120), Rotation(pitch=-0.032225, yaw=-0.061493, roll=-0.000488))

#  Transform(Location(x=6.447057, y=-29.880699, z=0.203184), Rotation(pitch=-0.062284, yaw=-0.075714, roll=-0.000275))


def Parking_two_static_vehicle(blueprint_library, world):
    #
    spawn_point = carla.Transform(carla.Location(x=9.506844, y=-27.037685, z=0.2),
                                  carla.Rotation(pitch=0.030114, yaw=-1.491394, roll=-0.001892))
    bp_vehicle = random.choice(blueprint_library.filter('vehicle.audi.tt'))
    other_vehicle_1 = world.try_spawn_actor(bp_vehicle, spawn_point)

    spawn_point = carla.Transform(carla.Location(x=9.487106, y=-32.604038, z=0.2),
                                  carla.Rotation(pitch=-0.021672, yaw=0.598833, roll=-0.000885))
    bp_vehicle = random.choice(blueprint_library.filter('vehicle.audi.tt'))
    other_vehicle_2 = world.try_spawn_actor(bp_vehicle, spawn_point)

    return other_vehicle_1, other_vehicle_2


def Parking_walk(blueprint_library, world):
    pedestrain_blueprints = world.get_blueprint_library().filter("walker.pedestrian.0001")
    pedestrain = world.try_spawn_actor(random.choice(pedestrain_blueprints),
                                       carla.Transform(carla.Location(x=14, y=-27.037685, z=0.2),
                                                       carla.Rotation(yaw=-96.712929)))

    return pedestrain



#这个文件是在动态的时候生成动态的车辆、行人这些的场景，carla_env只是定义了静态的场景
