from typing import Dict, Tuple, Text, Optional
import numpy as np
from highway_env.envs.common.observation import observation_factory
from highway_env.envs.common.action import action_factory
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.envs.common.graphics import EnvViewer
from highway_env.a_star import AStarPlanner

from collections import deque
import traceback

from KATECH_MAS_util import bezier_curve

class MAIntersectionEnv(AbstractEnv):
    AV_seq = []
    re_ttc = []
    re_speed = []
    astar_obs = []
    init_flag = True
    init_flag_reset = True
    ee_node = {'W':'E', 'E':'W', 'N':'S', 'S':'N'}

    def define_spaces(self):
        super().define_spaces()
        self.action = action_factory(self, self.config["action"])
        self.action_space = [self.action.space() for _ in range(self.config['controlled_vehicles'])]
        if "observation" not in self.config:
            raise ValueError("The observation configuration must be defined")
        self.observation = observation_factory(self, self.config["observation"])
        self.observation_space = [self.observation.space() for _ in range(self.config['controlled_vehicles'])]

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 2,
                "features": ["absx", "absy", "absvx", "absvy", "heading",
                              "x", "y", "vx", "vy",
                              "px1", "py1", "px2", "py2", "px3", "py3", "px4", "py4", "px5", "py5"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-10, 10],
                    "vy": [-10, 10],
                    "speed": [-10, 10],
                    "acceleration": [-1, 1],
                    "heading": [-np.pi/4, np.pi/4],
                    "absx": [-50, 50],
                    "absy": [-50, 50],
                    "absvx": [-5, 5],
                    "absvy": [-5, 5],
                },
                "absolute": False,
                "flatten": False,
                "normalize": True,
                "observe_intentions": False,
                "dynamical" : False,
                "grid_size" : [[-35,35], [-7.5,7.5]],
                "grid_step" : [10, 5]
            },
            "action": {
                "type": "ContinuousAction",
                "acceleration_range" : [-2, 2],
                "steering_range" : [-np.pi/4, np.pi/4],
                "longitudinal" : True,
                "lateral" : False,
            },
            "duration": 100,  # [s]
            "destination": "o3",
            "controlled_vehicles": 2,
            "initial_vehicle_count": 3,
            "spawn_probability": 1,
            "screen_width": 896,
            "screen_height": 896,
            "centering_position": [0.5, 0.5],
            "scaling": 4,
            "collision_reward": -20,
            "high_speed_reward": 1,
            "arrived_reward": 20,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "offroad_terminal": False,
            'offscreen_rendering': False
        })
        return config

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        reward_list = []
        for vehicle in self.vehicle:
            reward = 0
            reward += self.config["collision_reward"] * vehicle.crashed
            reward += self.speed_reward(vehicle)
            reward += self.config["arrived_reward"] * self.has_arrived(vehicle)
            reward_list.append([reward])
        if self.config["normalize_reward"]:
            reward = utils.remap(reward, [self.config["collision_reward"], self.config["arrived_reward"]], [0, 1])
        return reward_list

    def ttc_reward(self, vehicle):
        ttc_reward = 0
        v_front, v_rear = self.detect_vehicles(vehicle)
        threshold = 1
        if v_front is not None:
            if vehicle.speed is not v_front.speed:
                ttc = np.linalg.norm(vehicle.position - v_front.position) / (vehicle.speed - v_front.speed)
                if ttc > 0 and ttc < threshold:
                    ttc_reward += (ttc - threshold)
                else :
                    ttc_reward += 1
            else:
                ttc_reward += 1

        return ttc_reward
    
    def speed_reward(self, vehicle):
        speed_reward = 0
        if vehicle.speed < 0.5:
            speed_reward -= 2
        elif vehicle.speed < 6:
            speed_reward += vehicle.speed / 6
        else:
            speed_reward += - vehicle.speed / 2 + 4
        self.re_speed.append(speed_reward)
        return speed_reward
    
    def _is_terminated(self) -> bool:
        terminated_list = []
        crashed_list = []
        crashed = False
        for vehicle in self.vehicle:
            crashed = crashed or vehicle.crashed
            if crashed:
                for _ in range(len(self.vehicle)):
                    crashed_list.append([crashed])
                return crashed_list
            terminated = self.steps >= self.config["duration"] * self.config["policy_frequency"]\
                or self.has_arrived(vehicle)
            terminated_list.append([terminated])
        return terminated_list

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        return info

    def reset(self, map_data=None, vehs_data=None, geo_data=None, hv_data=None, destination=None, display=True):
        self.map_data = map_data
        self.vehs_data = vehs_data
        self.geo_data = geo_data
        self.hv_data = hv_data
        self.steps = 0
        self.ref_path = []
        
        if not display:
            self.default_config()
        try:
            if map_data != None:
                self.destination = destination + 'OR1'
                self._make_road()
                self._make_geofence()
                self.viewer = EnvViewer(self)
                self.viewer.display()
                self.init_image = self.viewer.get_image()
                self.global_mapping()
            if vehs_data != None:
                self._make_vehicles()
                #self.make_graph()
                self._plan_path()
                return 0        
        except Exception as e:
            print(e)
            traceback.print_exc()
            print('Reset Exception Occur')
            print()
            return None
        
    # def step(self, action):
    #     results = super(MAIntersectionEnv, self).step(action)
    #     self.steps += 1
    #     self.update_grid()
    #     self._clear_vehicles()

    #     new_obs = self.observation_type.observe()
    #     new_obs = np.array(new_obs)
    #     agent_obs = results[0]
    #     rewards = results[1]
    #     dones = results[2]
    #     info = results[3]

    #     return agent_obs, new_obs, rewards, dones, info
    
    def step(self, map_data=None, vehs_data=None, geo_data=None, hv_data=None, destination=None, display=True):
        self.map_data = map_data
        self.vehs_data = vehs_data
        self.geo_data = geo_data
        self.hv_data = hv_data
        self.steps = 0
        self.ref_path = []
        if not display:
            self.default_config()
        try:
            if map_data != None:
                self.destination = destination + 'OR1'
                self._make_road()
                self._make_geofence()
                #self.viewer = EnvViewer(self)
                self.viewer.display()
                self.init_image = self.viewer.get_image()
                self.global_mapping()
            if vehs_data != None:
                self._make_vehicles()
                self._plan_path()
                return 0        
            # self._simulate()
            # self._automatic_rendering()
            # self.steps += 1
        except Exception as e:
            print(e)
            traceback.print_exc()
            print('Reset Exception Occur')
            print()
            return None

    def _make_road(self) -> None:
        lane_gap: float = 0.05

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        net.add_lane('o0', 'ir0', StraightLane(self.map_data['SOR1_trfb']+lane_gap, self.map_data['SIR1_trfb'], width=self.map_data['lane_width_ns'], line_types=[c, c], priority=1, speed_limit=7))
        net.add_lane('o1', 'ir1', StraightLane(self.map_data['EOL1_trfb']+lane_gap, self.map_data['EIL1_trfb'], width=self.map_data['lane_width_we'], line_types=[c, s], priority=1, speed_limit=7))
        net.add_lane('o1', 'ir1', StraightLane(self.map_data['EOL2_trfb']+lane_gap, self.map_data['EIL2_trfb'], width=self.map_data['lane_width_we'], line_types=[s, c], priority=1, speed_limit=7))
        net.add_lane('o2', 'ir2', StraightLane(self.map_data['NOR1_trfb']+lane_gap, self.map_data['NIR1_trfb'], width=self.map_data['lane_width_ns'], line_types=[c, c], priority=1, speed_limit=7))
        net.add_lane('o3', 'ir3', StraightLane(self.map_data['WOL1_trfb']+lane_gap, self.map_data['WIL1_trfb'], width=self.map_data['lane_width_we'], line_types=[c, s], priority=1, speed_limit=7))
        net.add_lane('o3', 'ir3', StraightLane(self.map_data['WOL2_trfb']+lane_gap, self.map_data['WIL2_trfb'], width=self.map_data['lane_width_we'], line_types=[s, c], priority=1, speed_limit=7))

        net.add_lane('il0', 'o0', StraightLane(self.map_data['SIL1_trfb']-lane_gap, self.map_data['SOL1_trfb'], width=self.map_data['lane_width_ns'], line_types=[c, c], priority=1, speed_limit=7))
        net.add_lane('il1', 'o1', StraightLane(self.map_data['EIR1_trfb']-lane_gap, self.map_data['EOR1_trfb'], width=self.map_data['lane_width_we'], line_types=[c, s], priority=1, speed_limit=7))
        net.add_lane('il1', 'o1', StraightLane(self.map_data['EIR2_trfb']-lane_gap, self.map_data['EOR2_trfb'], width=self.map_data['lane_width_we'], line_types=[s, c], priority=1, speed_limit=7))
        net.add_lane('il2', 'o2', StraightLane(self.map_data['NIL1_trfb']-lane_gap, self.map_data['NOL1_trfb'], width=self.map_data['lane_width_ns'], line_types=[c, c], priority=1, speed_limit=7))
        net.add_lane('il3', 'o3', StraightLane(self.map_data['WIR1_trfb']-lane_gap, self.map_data['WOR1_trfb'], width=self.map_data['lane_width_we'], line_types=[c, s], priority=1, speed_limit=7))
        net.add_lane('il3', 'o3', StraightLane(self.map_data['WIR2_trfb']-lane_gap, self.map_data['WOR2_trfb'], width=self.map_data['lane_width_we'], line_types=[s, c], priority=1, speed_limit=7))

        net.add_lane('ir0', 'il2', StraightLane(self.map_data['SIR1_trfb']+lane_gap, self.map_data['NIL1_trfb'], width=self.map_data['lane_width_ns'], line_types=[n, n], priority=1, speed_limit=7))
        net.add_lane('ir1', 'il3', StraightLane(self.map_data['EIL1_trfb']+lane_gap, self.map_data['WIR1_trfb'], width=self.map_data['lane_width_we'], line_types=[n, n], priority=1, speed_limit=7))
        net.add_lane('ir1', 'il3', StraightLane(self.map_data['EIL2_trfb']+lane_gap, self.map_data['WIR2_trfb'], width=self.map_data['lane_width_we'], line_types=[n, n], priority=1, speed_limit=7))
        net.add_lane('ir2', 'il0', StraightLane(self.map_data['NIR1_trfb']+lane_gap, self.map_data['SIL1_trfb'], width=self.map_data['lane_width_ns'], line_types=[n, n], priority=1, speed_limit=7))
        net.add_lane('ir3', 'il1', StraightLane(self.map_data['WIL1_trfb']+lane_gap, self.map_data['EIR1_trfb'], width=self.map_data['lane_width_we'], line_types=[n, n], priority=1, speed_limit=7))
        net.add_lane('ir3', 'il1', StraightLane(self.map_data['WIL2_trfb']+lane_gap, self.map_data['EIR2_trfb'], width=self.map_data['lane_width_we'], line_types=[n, n], priority=1, speed_limit=7))
        

        right_turn_radius = 20
        left_turn_radius = 20
       
        angle = np.radians(0)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        r_center = rotation @ np.array([right_turn_radius+0.7, right_turn_radius+3.35])
        net.add_lane('ir0', 'il1', CircularLane(r_center, right_turn_radius, np.radians(200), np.radians(250), width=self.map_data['lane_width_we'], line_types=[n, c], priority=2, speed_limit=7))
        l_center = rotation @ np.array([-left_turn_radius+self.map_data['lane_width_ns']/2, left_turn_radius-self.map_data['lane_width_ns']/2])
        net.add_lane('ir0', 'il3', CircularLane(l_center, left_turn_radius, np.radians(0), np.radians(-90), width=self.map_data['lane_width_we'], clockwise=False, line_types=[n, n], priority=2, speed_limit=7))

        angle = np.radians(90)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        r_center = rotation @ np.array([right_turn_radius+3.35, right_turn_radius+0.7])
        net.add_lane('ir3', 'il0', CircularLane(r_center, right_turn_radius, np.radians(290), np.radians(340), width=self.map_data['lane_width_we'], line_types=[n, c], priority=2, speed_limit=7))
        l_center = rotation @ np.array([-left_turn_radius+self.map_data['lane_width_ns']/2, left_turn_radius-self.map_data['lane_width_ns']/2])
        net.add_lane('ir3', 'il2', CircularLane(l_center, left_turn_radius, np.radians(90), np.radians(0), width=self.map_data['lane_width_we'], clockwise=False, line_types=[n, n], priority=2, speed_limit=7))

        angle = np.radians(180)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        r_center = rotation @ np.array([right_turn_radius+0.9, right_turn_radius+3.9])
        net.add_lane('ir2', 'il3', CircularLane(r_center, right_turn_radius, np.radians(20), np.radians(70), width=self.map_data['lane_width_we'], line_types=[n, c], priority=2, speed_limit=7))
        l_center = rotation @ np.array([-left_turn_radius+self.map_data['lane_width_ns']/2, left_turn_radius-self.map_data['lane_width_ns']/2])
        net.add_lane('ir2', 'il1', CircularLane(l_center, left_turn_radius, np.radians(180), np.radians(90), width=self.map_data['lane_width_we'], clockwise=False, line_types=[n, n], priority=2, speed_limit=7))

        angle = np.radians(270)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        r_center = rotation @ np.array([right_turn_radius+3.9, right_turn_radius+0.9])
        net.add_lane('ir1', 'il2', CircularLane(r_center, right_turn_radius, np.radians(110), np.radians(160), width=self.map_data['lane_width_we'], line_types=[n, c], priority=2, speed_limit=7))
        l_center = rotation @ np.array([-left_turn_radius+self.map_data['lane_width_ns']/2, left_turn_radius-self.map_data['lane_width_ns']/2])
        net.add_lane('ir1', 'il0', CircularLane(l_center, left_turn_radius, np.radians(270), np.radians(180), width=self.map_data['lane_width_we'], clockwise=False, line_types=[n, n], priority=2, speed_limit=7))
        self.net = net
        self.road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        
    def _make_geofence(self):
        self.road.objects = []
        self.astar_obs = []
        
        Obstacle.LENGTH = 8
        Obstacle.WIDTH = 5
        self.geo_num = len(self.geo_data)
        self.pass_node = deque()
        for geo in self.geo_data.values():
            obs_pos = np.array([[geo['pos_tr'][0], geo['pos_tr'][1]],[geo['pos_tr'][0], geo['pos_tr'][1]]])
            self.road.objects.append(Obstacle(self.road, position=geo['pos_tr']))
            for obs in obs_pos:
                for ox in range(-(Obstacle.LENGTH + 5)//2 - 1, (Obstacle.LENGTH + 4)//2 + 1):
                    for oy in range(-(Obstacle.WIDTH + 5)//2 - 1, (Obstacle.WIDTH + 4)//2 + 1):
                        self.astar_obs.append([obs[0] + ox, obs[1] + oy])
            
            ## geofence가 중앙선 넘으면 통행불가 판단 ##################################################################################################
            x, y = geo['pos_tr']
            # Hit
            if y >= self.map_data['WIL2_trfb'][1]-self.map_data['lane_width_we'] and y <= self.map_data['WIL2_trfb'][1]+self.map_data['lane_width_we']:
                if y >= self.map_data['WIL2_trfb'][1]+0.5:
                    self.pass_node.append([x, self.map_data['WIL1_trfb'][1], 'WX'])
                else:
                    self.pass_node.append([x, self.map_data['WIL1_trfb'][1], 'W1'])
            elif y >= self.map_data['WIL1_trfb'][1]-self.map_data['lane_width_we'] and y <= self.map_data['WIL1_trfb'][1]+self.map_data['lane_width_we']:
                if y <= self.map_data['WIL1_trfb'][1]-0.5:
                    self.pass_node.append([x, self.map_data['WIL2_trfb'][1], 'WX'])
                else:
                    self.pass_node.append([x, self.map_data['WIL2_trfb'][1], 'W2'])
            elif y >= self.map_data['WIR1_trfb'][1]-self.map_data['lane_width_we'] and y <= self.map_data['WIR1_trfb'][1]+self.map_data['lane_width_we']:
                if y >= self.map_data['WIR1_trfb'][1]+0.5:
                    self.pass_node.append([x, self.map_data['WIR2_trfb'][1], 'EX'])
                else:
                    self.pass_node.append([x, self.map_data['WIR2_trfb'][1], 'E2'])
            elif y >= self.map_data['WIR2_trfb'][1]-self.map_data['lane_width_we'] and y <= self.map_data['WIR2_trfb'][1]+self.map_data['lane_width_we']:
                if y >= self.map_data['WIR2_trfb'][1]-0.5:
                    self.pass_node.append([x, self.map_data['WIR1_trfb'][1], 'EX'])
                else:
                    self.pass_node.append([x, self.map_data['WIR1_trfb'][1], 'E1'])

        
    def _make_vehicles(self) -> None:  
        self.vehicle = {}
        for id, veh in self.vehs_data.items():
            lane_id = self.road.network.get_closest_lane_index(veh['pos_tr'][-1])
            current_lane = self.road.network.get_lane(lane_id)
            vehicle = ControlledVehicle(self.road, 
                                        veh['pos_tr'][-1],
                                        speed=current_lane.speed_limit,
                                        heading=current_lane.heading_at(0))
            
            vehicle.type = veh['type']
            self.vehicle[id] = vehicle
            self.road.vehicles.append(vehicle)
        try:
            for id, hv in self.hv_data.items():     
                lane_id = self.road.network.get_closest_lane_index(hv['pos_tr'][-1])
                current_lane = self.road.network.get_lane(lane_id)
                vehicle = IDMVehicle(self.road, 
                            hv['pos_tr'][-1],
                            speed=current_lane.speed_limit,
                            heading=current_lane.heading_at(0))
                vehicle.type = 'HV'
                self.vehicle[id] = vehicle
                self.road.vehicles.append(vehicle)
        except:
            pass
    
    def _plan_path(self):
        bezier_cp = []
        x, y, dir = self.pass_node[0]
        if dir[1] != 'X':
            if dir == 'W1':
                cp_x = np.array([-15, -10, -5, 0, 10])
                cp_y = np.array([3.5, 3.5, 0, -0.25, 0])
            elif dir == 'W2':
                cp_x = np.array([-15, -10, -5, 0, 10])
                cp_y = np.array([-3.5, -3.5, 0, 0.25, 0])
            elif dir == 'E1':
                cp_x = np.array([15, 10, 5, 0, -10])
                cp_y = np.array([-3.5, -3.5, 0, 0.25, 0])
            elif dir == 'E2':  
                cp_x = np.array([15, 10, 5, 0, -10])
                cp_y = np.array([3.5, 3.5, 0, -0.25, 0])
                
            bezier_cp += [[x+cp_x[0], y+cp_y[0]],
                          [x+cp_x[1], y+cp_y[1]], 
                          [x+cp_x[2], y+cp_y[2]], 
                          [x+cp_x[3], y+cp_y[3]],
                          [x+cp_x[4], y+cp_y[4]]]
            if bezier_cp:
                self.ref_path = bezier_curve(np.linspace(0, 1, 50), np.array(bezier_cp))

        # AV priority
        self._determine_priority()
        
        # AV path random
        ext_goal_array = np.linspace(0, 20, len(self.vehicle))

        # # AV availability & AV path
        for i, (id, veh) in enumerate(self.vehs_data_ordered):
            if veh['priority'] == -1:
                continue
            try:
                av = self.vehicle[id]
                for x, y, dir in self.pass_node:
                    if self.destination[0] == self.ee_node[dir[0]]: # geofence 방향이랑 passing 방향이랑 같으니까
                        if dir[1] != 'X':
                            veh['availability'] = 1
                            self.global_planning(av, ext_goal_array[i])
                            if id != 1: #if HDV about 5 second path
                                av.global_path = av.global_path[:10]
                        else:
                            veh['availability'] = 2
                            av.global_path = np.array([])
                    else:
                        veh['availability'] = 0
                        self.global_planning(av, ext_goal_array[i])
                    veh['path_tr'] = np.array(av.global_path)
            except Exception as e:
                veh['availability'] = 1
                veh['path_tr'] = np.array([])
                print(e)
                traceback.print_exc()
                print('AV id', id, 'Cannot find path')
                print()

    def _determine_priority(self):
        x, y, dir = self.pass_node[0]
        ordered_veh = {}
        for id, veh in self.vehs_data.items():
            if (dir[0] == 'W') and (veh['pos_tr'][-1][0] < (x+10)) and (veh['pos_tr'][-1][1] > 0) and (veh['pos_tr'][-1][1] < self.map_data['SIR1_trfb'][1]) or \
               (dir[0] == 'E') and (veh['pos_tr'][-1][0] > (x-10)) and (veh['pos_tr'][-1][1] < 0) and (veh['pos_tr'][-1][1] > self.map_data['NIR1_trfb'][1]):
                ordered_veh[id] = np.sum(np.abs(np.array([x - veh['pos_tr'][-1][0], np.abs(y - veh['pos_tr'][-1][1])*1.8])))
            else:
                veh['priority'] = -1
                veh['availability'] = 0
                veh['path_tr'] = np.array([])
        # # HV도 고려해서 priority 결정할 경우
        # for id, hv in self.hv_data.items():
        #     if hv['pos_tr'][-1][0] <= x:
        #         ordered_veh[id] = np.sum(np.abs((np.array([x, y+4])-hv['pos_tr'])))
        #     else:
        #         hv['priority'] = -1
        
        ordered_veh = sorted(ordered_veh.items(), key=lambda x:x[1])
        for i, (id, _) in enumerate(ordered_veh):
            self.vehs_data[id]['priority'] = i + 1
        self.vehs_data_ordered = sorted(self.vehs_data.items(), key=lambda x:x[1]['priority'])

        
    def _clear_vehicles(self) -> None:
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.destination[0] - vehicle.position[0] < 35
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              not (is_leaving(vehicle) or vehicle.route is None)]
        self.vehicle = [vehicle for vehicle in self.vehicle if
                              not (is_leaving(vehicle) or vehicle.route is None or vehicle.crashed)]
        self.make_graph()
    
    def has_arrived(self, vehicle, exit_distance: float = 35) -> bool:
        arrived = True
        arrived = arrived \
            and vehicle.destination[0] - vehicle.position[0] < exit_distance
        return arrived
    
    @property
    def crashed(self):
        crashed = False
        for vehicle in self.vehicle:
            crashed = crashed or vehicle.crashed
        return crashed
    
    def detect_vehicles(self, vehicle):
        x_vehicle, y_vehicle = vehicle.position[0], vehicle.position[1]
        x_front = x_rear = None
        v_front = v_rear = None
        for v in self.road.vehicles:
            x_v, y_v = v.position[0], v.position[1]
            if v is not vehicle:
                if x_vehicle <= x_v and abs(y_vehicle - y_v) < 2 and (x_front is None or x_v <= x_front):
                    x_front = x_v
                    v_front = v
                if x_vehicle > x_v and abs(y_vehicle - y_v) < 2 and (x_rear is None or x_v > x_rear):
                    x_rear = x_v
                    v_rear = v
        return v_front, v_rear
    
    def global_mapping(self):
        self.ox = []
        self.oy = []
        scaling = self.config["scaling"]
        self.grid_size = 1

        gray_image = np.dot(self.init_image, [0.299, 0.587, 0.114])
        for i in range(self.config['screen_height']):
            for j in range(self.config['screen_width']):
                if gray_image[i][j] > 177 and gray_image[i][j] < 180:
                    gray_image[i][j] = 255
                    self.ox.append((j - len(gray_image[i])/2) / scaling)
                    self.oy.append((len(gray_image)/2 - i) / scaling)
                else:
                    gray_image[i][j] = 0

        min_x = min(self.ox)
        min_y = min(self.oy)
        max_x = max(self.ox)
        max_y = max(self.oy)
        x_width = round((max_x - min_x) / self.grid_size)
        y_width = round((max_y - min_y) / self.grid_size)

        self.obstacle_map = np.zeros((x_width, y_width), dtype=bool)

        for iox, ioy in zip(self.ox, self.oy):
            xf = round((iox - min_x) / self.grid_size)
            yf = round((ioy - min_y) / self.grid_size)
            if xf >= x_width:
                xf = x_width - 1
            elif yf >= y_width:
                yf = y_width - 1
            self.obstacle_map[xf][yf] = True

        for obs_pos in self.astar_obs:
            obs_x = round((obs_pos[0] - min_x) / self.grid_size)
            obs_y = round((obs_pos[1] - min_y) / self.grid_size)
            self.obstacle_map[obs_x][-obs_y] = True

    def global_planning(self, vehicle, var=15):
        path = []
        start_pos = np.array([vehicle.position[0], -vehicle.position[1]])    

        px, py, dir = self.pass_node[0]
        if dir[0] == 'W': 
            px_var = px - var
            goal_x = px + 20
            py_var = py.copy()
            goal_y = py.copy()
        elif dir[0] == 'E':
            px_var = px + var
            goal_x = px - 20
            py_var = py.copy()
            goal_y = py.copy()
        elif dir[0] == 'N':
            px_var = px.copy()
            goal_x = px.copy()
            py_var = py + var
            goal_y = py - 20
        elif dir[0] == 'S':
            px_var = px.copy()
            goal_x = px.copy()
            py_var = py - var
            goal_y = py + 20
        a_star = AStarPlanner(self.ox, self.oy, self.obstacle_map, self.grid_size)
        rx, ry = a_star.planning(start_pos[0], start_pos[1], px_var, -py_var)
        for x in range(len(rx)-1, -1, -4):
            if [rx[x], -ry[x]] not in path:
                path.append([rx[x], -ry[x]])

        if len(path) >= 6:
            if dir[0] == 'W':
                ex = np.arange(path[-1][0], goal_x, 1)
                ey = np.full(ex.shape, goal_y)
                ep = np.hstack([ex[:, np.newaxis], ey[:, None]])
                total_path = np.concatenate((np.array(path), ep))
            elif  dir[0] == 'E':
                ex = np.arange(goal_x, path[-1][0], -1)
                ey = np.full(ex.shape, goal_y)
                ep = np.hstack([ex[:, np.newaxis], ey[:, None]])
                total_path = np.concatenate((ep, np.array(path)))
            elif  dir[0] == 'S':
                ey = np.arange(path[-1][0], goal_y, 1)
                ex = np.full(ey.shape, goal_x)
                ep = np.hstack([ex[:, None], ey[:, np.newaxis]])
                total_path = np.concatenate((np.array(path), ep))
            elif  dir[0] == 'N':
                ey = np.arange(goal_y, path[-1][0], -1)
                ex = np.full(ey.shape, goal_x)
                ep = np.hstack([ex[:, None], ey[:, np.newaxis]])
                total_path = np.concatenate((ep, np.array(path)))
            total_path = np.array(total_path) + np.random.randn(1)/5
            vehicle.global_path = bezier_curve(np.linspace(0, 1, 50), total_path)
        else:
            vehicle.global_path = []
        vehicle.grid = self.make_grid(vehicle)
        vehicle.forward_path = vehicle.grid[0:5]
    
    def make_grid(self, vehicle):
        grid = []
        path = np.array(vehicle.global_path)
        for pt in path:
            if abs(pt[0]) <= 8 and abs(pt[1]) <= 8:
                pt = pt // np.array([4, 4])
            else:
                if abs(pt[0]) > abs(pt[1]):
                    pt = pt // np.array([5, 4])
                elif abs(pt[0]) < abs(pt[1]):
                    pt = pt // np.array([4, 5])
                else:
                    pt = pt // np.array([5, 5])
            if pt.tolist() not in grid:
                grid.append(pt.tolist())
        return grid
    
    def make_graph(self):
        self.graph = []
        for target in self.vehicle:
            target.graph = []
            if target not in target.graph:
                target.graph.append(target)
            for neighbor in self.road.vehicles:
                if neighbor is not target:
                    if self.is_intersect(target, neighbor):
                        target.graph.append(neighbor)
            self.graph.append(target.graph)
    
    def is_intersect(self, v1, v2):
        for idx in v2.grid:
            if idx in v1.grid:
                return True
        return False
    
    def update_grid(self):
        for vehicle in self.vehicle:
            pt = vehicle.lookahead_point
            if abs(pt[0]) <= 8 and abs(pt[1]) <= 8:
                    pt = pt // np.array([4, 4])
            else:
                if abs(pt[0]) > abs(pt[1]):
                    pt = pt // np.array([5, 4])
                elif abs(pt[0]) < abs(pt[1]):
                    pt = pt // np.array([4, 5])
                else:
                    pt = pt // np.array([5, 5])
            idx = vehicle.grid.index(pt.tolist())
            vehicle.grid = vehicle.grid[idx:]
            vehicle.forward_path = vehicle.grid[0:5]