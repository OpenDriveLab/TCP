import glob
import os
import sys
import lxml.etree as ET
import argparse
import random
import time

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


WEATHER_LIST = {'ClearNoon':{'cloudiness': 15.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 0.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 0.0,
                             'sun_altitude_angle': 75.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'CloudyNoon':{'cloudiness': 80.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 0.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 45.0,
                             'sun_altitude_angle': 75.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'WetNoon':{'cloudiness': 20.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 45.0,
                             'sun_altitude_angle': 75.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'WetCloudyNoon':{'cloudiness': 90.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 180.0,
                             'sun_altitude_angle': 75.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'SoftRainNoon':{'cloudiness': 90.0,
                             'precipitation': 15.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 315.0,
                             'sun_altitude_angle': 75.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'MidRainyNoon':{'cloudiness': 80.0,
                             'precipitation': 30.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.40,
                             'sun_azimuth_angle': 0.0,
                             'sun_altitude_angle': 75.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'HardRainNoon':{'cloudiness': 90.0,
                             'precipitation': 60.0,
                             'precipitation_deposits': 100.0,
                             'wind_intensity':1.0,
                             'sun_azimuth_angle': 90.0,
                             'sun_altitude_angle': 75.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'ClearSunset':{'cloudiness': 15.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 0.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 45.0,
                             'sun_altitude_angle': 15.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'CloudySunset':{'cloudiness': 80.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 0.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 270.0,
                             'sun_altitude_angle': 15.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'WetSunset':{'cloudiness': 20.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 270.0,
                             'sun_altitude_angle': 15.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'WetCloudySunset':{'cloudiness': 90.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 0.0,
                             'sun_altitude_angle': 15.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'MidRainSunset':{'cloudiness': 80.0,
                             'precipitation': 30.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.40,
                             'sun_azimuth_angle': 270.0,
                             'sun_altitude_angle': 15.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'HardRainSunset':{'cloudiness': 80.0,
                             'precipitation': 60.0,
                             'precipitation_deposits': 100.0,
                             'wind_intensity':1.0,
                             'sun_azimuth_angle': 0.0,
                             'sun_altitude_angle': 15.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'SoftRainSunset':{'cloudiness': 90.0,
                             'precipitation': 15.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 270.0,
                             'sun_altitude_angle': 15.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'ClearNight':{'cloudiness': 15.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 0.0,
                             'wind_intensity': 0.35,
                             'sun_azimuth_angle': 0.0,
                             'sun_altitude_angle': -80.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'CloudyNight':{'cloudiness': 80.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 0.0,
                             'wind_intensity': 0.35,
                             'sun_azimuth_angle': 45.0,
                             'sun_altitude_angle': -80.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'WetNight':{'cloudiness': 20.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity': 0.35,
                             'sun_azimuth_angle': 225.0,
                             'sun_altitude_angle': -80.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'WetCloudyNight':{'cloudiness': 90.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity': 0.35,
                             'sun_azimuth_angle': 225.0,
                             'sun_altitude_angle': -80.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'SoftRainNight':{'cloudiness': 90.0,
                             'precipitation': 15.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity': 0.35,
                             'sun_azimuth_angle': 270.0,
                             'sun_altitude_angle': -80.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'MidRainyNight':{'cloudiness': 80.0,
                             'precipitation': 30.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.4,
                             'sun_azimuth_angle': 225.0,
                             'sun_altitude_angle': -80.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'HardRainNight':{'cloudiness': 90.0,
                             'precipitation': 60.0,
                             'precipitation_deposits': 100.0,
                             'wind_intensity':1,
                             'sun_azimuth_angle': 225.0,
                             'sun_altitude_angle': -80.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                }





def interpolate_trajectory(world_map, waypoints_trajectory, hop_resolution=1.0):
    """
    Given some raw keypoints interpolate a full dense trajectory to be used by the user.
    Args:
        world: an reference to the CARLA world so we can use the planner
        waypoints_trajectory: the current coarse trajectory
        hop_resolution: is the resolution, how dense is the provided trajectory going to be made
    Return: 
        route: full interpolated route both in GPS coordinates and also in its original form.
    """

    dao = GlobalRoutePlannerDAO(world_map, hop_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    # Obtain route plan
    route = []
    is_junction = False
    distance = 0

    try:
        for i in range(len(waypoints_trajectory) - 1):   # Goes until the one before the last.

            waypoint = waypoints_trajectory[i]
            waypoint_next = waypoints_trajectory[i + 1]
            interpolated_trace = grp.trace_route(waypoint, waypoint_next)
            for i, wp_tuple in enumerate(interpolated_trace):
                route.append(wp_tuple[0].transform)
                if i > 0:
                    distance += wp_tuple[0].transform.location.distance(interpolated_trace[i-1][0].transform.location)
                if not is_junction:
                    is_junction = wp_tuple[0].is_junction
                # print (wp_tuple[0].transform.location, wp_tuple[1])
    except:
        return None, distance, is_junction

    return route, distance, is_junction


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    world = client.load_world(args.town)
    print ('loaded world')

    waypoint_list = world.get_map().generate_waypoints(2.0)
    print ('got %d possible waypoint'%len(waypoint_list))

    actors = world.get_actors()
    traffic_lights_list = actors.filter('*traffic_light')
    print ('got %d traffic lights'%len(traffic_lights_list))

    max_dist = 300
    min_dist = 50

    valid_route_list = []
    number = 0

    distance_list = []
    is_junction_list = []

    world_map = world.get_map()

    root = ET.Element('routes')
    while number < args.route_num:
        start_wp = random.choice(waypoint_list)
        end_wp = random.choice(waypoint_list)
        start_wp = world_map.get_waypoint(start_wp.transform.location,project_to_road=True, lane_type=carla.LaneType.Driving)
        end_wp = world_map.get_waypoint(end_wp.transform.location,project_to_road=True, lane_type=carla.LaneType.Driving)

        if start_wp.transform.location.distance(end_wp.transform.location) < min_dist or start_wp.transform.location.distance(end_wp.transform.location) > max_dist:
            continue

        wp_list, distance, is_junction = interpolate_trajectory(world_map, [start_wp.transform.location, end_wp.transform.location])
        if wp_list is None:
            continue
        if distance <= min_dist or distance >= max_dist:
            continue
        distance_list.append(distance)
        is_junction_list.append(is_junction)

        route = ET.SubElement(root, 'route', id='%d'%number, town=args.town)

        # choose a random weather
        weather = random.choice(list(WEATHER_LIST.keys()))
        route = ET.SubElement(route, 'weather', id=weather, cloudiness='%f'%WEATHER_LIST[weather]['cloudiness'], precipitation='%f'%WEATHER_LIST[weather]['precipitation'], precipitation_deposits='%f'%WEATHER_LIST[weather]['precipitation_deposits'], wind_intensity='%f'%WEATHER_LIST[weather]['wind_intensity'], sun_azimuth_angle='%f'%WEATHER_LIST[weather]['sun_azimuth_angle'], sun_altitude_angle='%f'%WEATHER_LIST[weather]['sun_altitude_angle'], fog_density='%f'%WEATHER_LIST[weather]['fog_density'], fog_distance='%f'%WEATHER_LIST[weather]['fog_distance'], fog_falloff='%f'%WEATHER_LIST[weather]['fog_falloff'],wetness='%f'%WEATHER_LIST[weather]['wetness'])

        x, y, yaw = start_wp.transform.location.x, start_wp.transform.location.y, start_wp.transform.rotation.yaw
        ET.SubElement(route, 'waypoint', x='%f'%x, y='%f'%y, z='0.0', 
                                            pitch='0.0', roll='0.0', yaw='%f'%yaw)
        x, y, yaw = end_wp.transform.location.x, end_wp.transform.location.y, end_wp.transform.rotation.yaw
        ET.SubElement(route, 'waypoint', x='%f'%x, y='%f'%y, z='0.0', 
                                            pitch='0.0', roll='0.0', yaw='%f'%yaw)
        
        number += 1
        print(number)

    print('Avg distance:', sum(distance_list)/number)
    print('Portion of going through junction:', sum(is_junction_list)/number)

    tree = ET.ElementTree(root)

    if args.save_file is not None:
        tree.write(args.save_file, xml_declaration=True, encoding='utf-8', pretty_print=True)

if __name__ == '__main__':
    global args

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_file', type=str, required=False, default="/home/wupenghao/transfuser/leaderboard/data/extra_routes/routes_town10_1000k.xml", help='xml file path to save the route waypoints')
    parser.add_argument('--town', type=str, default='Town10HD', help='town for generating routes')
    parser.add_argument('--route_num', type=int, default=1200, help='number of routes')
    
    args = parser.parse_args()

    main()
