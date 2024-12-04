"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math
import time
import matplotlib.pyplot as plt

show_animation = True


class AStarPlanner:

    def __init__(self, ox, oy, obstacle_map, resolution):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.motion = self.get_motion_model()
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.obstacle_map = obstacle_map

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)
        open_set_s, closed_set_s = dict(), dict()
        open_set_s[self.calc_grid_index(start_node)] = start_node
        open_set_g, closed_set_g = dict(), dict()
        open_set_g[self.calc_grid_index(goal_node)] = goal_node

        while open_set_s and open_set_g:

            c_id_s = min(open_set_s,
                key=lambda o: open_set_s[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set_s[o]))
            current_s = open_set_s[c_id_s]
            '''
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current_s.x, self.min_x),
                         self.calc_grid_position(current_s.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set_s.keys()) % 10 == 0:
                    plt.pause(0.001)
            '''

            del open_set_s[c_id_s]

            closed_set_s[c_id_s] = current_s

            if c_id_s in closed_set_g:
                meet_node = closed_set_g[c_id_s]
                closed_set_s[meet_node.parent_index] = self.Node(current_s.x, current_s.y, current_s.cost, c_id_s)
                break

            for i, _ in enumerate(self.motion):
                node = self.Node(current_s.x + self.motion[i][0],
                                 current_s.y + self.motion[i][1],
                                 current_s.cost + self.motion[i][2], c_id_s)
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue

                if n_id in closed_set_s:
                    continue

                if n_id not in open_set_s:
                    open_set_s[n_id] = node
                else:
                    if open_set_s[n_id].cost > node.cost:
                        open_set_s[n_id] = node

            c_id_g = min(open_set_g,
                key=lambda o: open_set_g[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set_g[o]))
            current_g = open_set_g[c_id_g]
            
            

            del open_set_g[c_id_g]

            closed_set_g[c_id_g] = current_g

            if c_id_g in closed_set_s:
                meet_node = closed_set_s[c_id_g]
                closed_set_g[meet_node.parent_index] = self.Node(current_g.x, current_g.y, current_g.cost, c_id_g)
                break

            for i, _ in enumerate(self.motion):
                node = self.Node(current_g.x + self.motion[i][0],
                                 current_g.y + self.motion[i][1],
                                 current_g.cost + self.motion[i][2], c_id_g)
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue

                if n_id in closed_set_g:
                    continue

                if n_id not in open_set_g:
                    open_set_g[n_id] = node
                else:
                    if open_set_g[n_id].cost > node.cost:
                        open_set_g[n_id] = node

        rx, ry = self.calc_final_path(meet_node, closed_set_s, closed_set_g)

        return rx, ry

    def calc_final_path(self, meet_node, closed_set_s, closed_set_g):
        rx, ry = [], []
        current = meet_node

        while current.parent_index != -1:
            rx.append(self.calc_grid_position(current.x, self.min_x))
            ry.append(self.calc_grid_position(current.y, self.min_y))
            current = closed_set_s[current.parent_index]

        rx.append(self.calc_grid_position(current.x, self.min_x))
        ry.append(self.calc_grid_position(current.y, self.min_y))

        rx.reverse()
        ry.reverse()

        current = meet_node
        if current.parent_index in closed_set_g:
            current = closed_set_g[current.parent_index]

        while current.parent_index != -1:
            rx.append(self.calc_grid_position(current.x, self.min_x))
            ry.append(self.calc_grid_position(current.y, self.min_y))
            current = closed_set_g[current.parent_index]

        rx.append(self.calc_grid_position(current.x, self.min_x))
        ry.append(self.calc_grid_position(current.y, self.min_y))

        rx.reverse()
        ry.reverse()
        return rx, ry
    
    @staticmethod
    def calc_heuristic(n1, n2):
        w = 0.5  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y) # 유클리드 거리
        #d = w * max(abs(n1.x - n2.x), abs(n1.y - n2.y)) # 체비셰프 거리
        #d = w * (abs(n1.x - n2.x) + abs(n1.y - n2.y)) # 맨허튼거리
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        try:
            if self.obstacle_map[node.x][node.y]:
                return False
        except:
            # print(node.x, node.y, self.obstacle_map.shape)
            #print('astar collision')
            pass

        return True
    
    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion