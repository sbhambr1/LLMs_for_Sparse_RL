import cv2
import gym
from addict import addict
from gym import spaces
import os
import numpy as np
import plotly.colors as colors
from skimage.transform import rescale, resize

from utils.grid_renderer import Grid_Renderer


class Env_Craft:
    def __init__(self, use_img_state=False):
        self.use_img_state = use_img_state
        # actions: up, down, left, right
        self.action_space = spaces.Discrete(4)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.observation_space = spaces.Box(low=0, high=1, shape=(154, ), dtype=np.float16)
        self.obs_type = np.int16
        self.success_reward = 0
        # object config
        self.height = 10
        self.width = 15
        self.objects = addict.Dict({
            'agent': {
                'id': 1,
                'location': (1, 1),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[1])  # red
            },
            'wall': {
                'id': 2,
                'color': colors.hex_to_rgb(colors.qualitative.T10[-1])  # dark brown
            },
            'wood': {
                'id': 3,
                'locations': [(3, 6), [1, 13]],
                'color': colors.hex_to_rgb(colors.qualitative.D3[5])  # dark brown
            },
            'workshop1': {
                'id': 4,
                'location': (7, 1),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[3])  # purple
            },
            'workshop2': {
                'id': 5,
                'location': (7, 3),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[2])  # green
            },
            'workshop3': {
                'id': 6,
                'location': (7, 5),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[0])  # blue
            },
            'wood_process': {
                'id': 7,
                'location': (7, 8),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[-1])  # yellow
            },

        })
        self.n_objects = len(self.objects)
        self.grid = np.zeros(shape=(self.height, self.width))
        self.carry_list = {'wood': 0}
        self.n_processed_wood = 0
        self.wood_collected = False
        self.all_wood_collected = False
        self.wood_early_stop = False
        # flags
        self.is_stick_made = False
        self.is_plank_made = False
        self.is_ladder_made = False
        self.at_workshop_1 = False
        self.at_workshop_2 = False
        self.at_workshop_3 = False
        self.at_wood_process = False
        # state
        self.info = dict()
        self.agent_pos = self.objects.agent.location
        self.workshop_locations = [self.objects[obj].location for obj in self.objects if 'workshop' in obj]
        self.locations_2_workshop = {self.objects[obj].location:self.objects[obj].id for obj in self.objects if 'workshop' in obj}
        # init
        self._init()
        # init renderer
        color_map = {self.objects[obj].id:self.objects[obj].color for obj in self.objects}
        self.renderer = Grid_Renderer(grid_size=20, color_map=color_map)

    def _init(self):
        # update flags and state info
        self.count = 0
        self.max_steps = 1500
        self.is_stick_made = False
        self.is_plank_made = False
        self.is_ladder_made = False
        self.at_workshop_1 = False
        self.at_workshop_2 = False
        self.at_workshop_3 = False
        self.at_wood_process = False
        self.carry_list = {'wood': 0}
        self.n_processed_wood = 0
        self.wood_collected = False
        self.all_wood_collected = False
        self.info = dict()
        self.agent_pos = self.objects.agent.location
        # init grid
        self.grid = np.zeros(shape=(self.height, self.width))
        # init wall
        self.grid[0:self.height, 0] = self.objects.wall.id
        self.grid[0:self.height, self.width-1] = self.objects.wall.id
        self.grid[0, 0:self.width] = self.objects.wall.id
        self.grid[self.height-1, 0:self.width] = self.objects.wall.id
        # place the agent
        self.grid[self.agent_pos[0], self.agent_pos[1]] = self.objects.agent.id
        # place the wood
        for wood_pos in self.objects.wood.locations:
            self.grid[wood_pos[0], wood_pos[1]] = self.objects.wood.id
        # place the workshop
        for workshop_pos in self.workshop_locations:
            self.grid[workshop_pos[0], workshop_pos[1]] = self.locations_2_workshop[workshop_pos]
        # place the wood process station
        self.grid[self.objects.wood_process.location[0], self.objects.wood_process.location[1]] = self.objects.wood_process.id

    def render(self, mode='rgb_array'):
        return self.renderer.render_2d_grid(self.grid)

    def _get_img_obs(self):
        rgb_img = self.render()
        obs = np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY), axis=2)
        obs = resize(obs, (84, 84), anti_aliasing=False)
        # to Pytorch channel format
        obs = np.moveaxis(obs, -1, 0) / 255.0
        return obs

    def _get_grid_obs(self):
        grid_obs = np.copy(self.grid).flatten()
        grid_obs = np.append(grid_obs, [int(self.is_stick_made), int(self.is_plank_made), int(self.is_ladder_made),
                                        self.n_processed_wood])
        return grid_obs.astype(self.obs_type)

    def _obs(self):
        if self.use_img_state:
            return self._get_img_obs()
        else:
            return self._get_grid_obs()

    def step(self, action):
        self.count += 1
        def _go_to(x, y):
            old_pos = tuple(self.agent_pos)
            self.agent_pos = (x, y)
            self.grid[x, y] = self.objects.agent.id
            if self.agent_pos != old_pos:
                if (old_pos[0], old_pos[1]) in self.workshop_locations:
                    self.grid[old_pos[0], old_pos[1]] = self.locations_2_workshop[(old_pos[0], old_pos[1])]
                elif (old_pos[0], old_pos[1]) == self.objects.wood_process.location:
                    self.grid[old_pos[0], old_pos[1]] = self.objects.wood_process.id
                else:
                    self.grid[old_pos[0], old_pos[1]] = 0

        assert action <= 3
        next_x, next_y = self.agent_pos[0] + self.actions[action][0], self.agent_pos[1] + self.actions[action][1]
        # update grid
        self.at_workshop_1 = False
        self.at_workshop_2 = False
        self.at_workshop_3 = False
        self.at_wood_process = False
        early_stop_done = False
        # check if the agent will hit into walls, then make no change
        if self.grid[next_x, next_y] == self.objects.wall.id:
            pass
        elif self.grid[next_x, next_y] == self.objects.wood.id:
            self.carry_list['wood'] += 1
            _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.workshop1.id:
            self.at_workshop_1 = True
            if self.n_processed_wood > 0 and not self.is_stick_made:
                self.is_stick_made = True
                self.n_processed_wood -= 1
            else:
                self.at_workshop_1 = False
                early_stop_done = False
            _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.workshop2.id:
            self.at_workshop_2 = True
            if self.n_processed_wood > 0 and not self.is_plank_made:
                self.is_plank_made = True
                self.n_processed_wood -= 1
            else:
                self.at_workshop_2 = False
                early_stop_done = False
            _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.workshop3.id:
            self.at_workshop_3 = True
            if self.is_plank_made and self.is_stick_made and not self.is_ladder_made:
                self.is_ladder_made = True
            else:
                self.at_workshop_3 = False
                early_stop_done = False
            _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.wood_process.id:
            self.at_wood_process = True
            if self.carry_list['wood'] > 0:
                if self.carry_list['wood'] != 2 and self.wood_early_stop:
                    early_stop_done = False
                # process raw wood
                #if not self.wood_collected:
                if not self.all_wood_collected:
                    #self.wood_collected = True
                    self.n_processed_wood += self.carry_list['wood']
                    self.carry_list['wood'] -= 1
                    if self.n_processed_wood == 2:
                        self.all_wood_collected = True
                else:
                    # not allow visiting the processing workshop twice
                    early_stop_done = False
            else:
                # not allow visiting the processing workshop with empty hand
                early_stop_done = False
            _go_to(next_x, next_y)
        else:
            _go_to(next_x, next_y)
        # update info
        self.info['at_workshop_1'] = int(self.at_workshop_1)
        self.info['at_workshop_2'] = int(self.at_workshop_2)
        self.info['at_workshop_3'] = int(self.at_workshop_3)
        self.info['at_wood_process'] = int(self.at_wood_process)
        self.info['wood_collected'] = int(self.wood_collected)
        self.info['all_wood_collected'] = int(self.all_wood_collected)
        self.info['n_processed_wood'] = int(self.n_processed_wood)
        self.info['carried_wood'] = self.carry_list['wood']
        self.info['is_stick_made'] = int(self.is_stick_made)
        self.info['is_plank_made'] = int(self.is_plank_made)
        self.info['is_ladder_made'] = int(self.is_ladder_made)
        self.info['next_tuple_state'] = tuple(self._get_grid_obs().tolist())
        if self.count >= self.max_steps:
            truncated = True
        else:
            truncated = False
        done = early_stop_done or self.is_ladder_made or truncated
        reward = self.success_reward if self.is_ladder_made else 0

        return self._obs(), float(reward), done, addict.Dict(self.info)

    def reset(self, **kwargs):
        self._init()
        return self._obs()


def main():
    import cv2
    env = Env_Craft()
    done = False

    while not done:
        grid_img = env.render()
        cv2.imshow('grid render', cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
        a = cv2.waitKey(0)
        # actions: up, down, left, right
        if a == ord('q'):
            break
        elif a == ord('w'):
            a = 0
        elif a == ord('a'):
            a = 2
        elif a == ord('s'):
            a = 1
        elif a == ord('d'):
            a = 3
        elif a == ord('e'):
            a = 4
        obs, reward, done, info = env.step(int(a))
        print(reward, done, info)


if __name__ == "__main__":
    main()
