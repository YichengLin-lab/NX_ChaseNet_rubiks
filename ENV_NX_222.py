import pickle
import random
import gym
from gym import spaces
import os
import cv2
import numpy as np
import gc
import math
from copy import deepcopy
import torch


def get_Dist(ori_obs, tar_obs, model, device):
    model.eval()
    with torch.no_grad():
        ori_obs = torch.tensor(ori_obs, dtype=torch.long).unsqueeze(0).to(device)
        tar_obs = torch.tensor(tar_obs, dtype=torch.long).unsqueeze(0).to(device)
        output = torch.round(model(ori_obs, tar_obs))
        
        if output.item() < 0:
            output = torch.tensor(0)
    
    return output.item()


class RubiksCube222Env(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'ansi']}

    def __init__(self):
        self.cube = None
        self.cube_reduced = None
        self.cube_state = None
        self.scramble_this_time = None
        
        self.ori_cube = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype=np.int8)
        self.now_cube = None
        # spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = np.ones(24)

        self.plate_dict = {}
        self.move_len = 0
        self.action_last = None
        self.s_continue = 0

        self.if_nx_module = False
        self.nx_module = None
        

        state_file = "rubiks_cube_222_states_FRU.pickle"

        with open(state_file, "rb") as f:
            self.cube_states = pickle.load(f)

    def set_nx_ready(self, nx_module):
        self.if_nx_module = True
        self.nx_module = nx_module

    def update_cube_reduced(self):
        self.cube_reduced = ''.join(TILE_MAP[tile] for tile in self.cube)

    def update_cube_state(self): 
        self.cube_state = self.cube_states[self.cube_reduced]

    def generate_scramble(self):
        scramble_len = 0
        prev_move = None
        scramble = ""
        moves = ['F', 'R', 'U']
        move_type = ['', '2', "'"]

        while scramble_len <= 14:
            move = random.choice(moves)
            while move == prev_move:
                move = random.choice(moves)
            scramble += move + random.choice(move_type) + " "
            prev_move = move
            scramble_len += 1

        return scramble[:-1]

    def move(self, move_side, move_type=None):
        repetitions = dict({None: 1, "2": 2, "'": 3})[move_type]

        if move_side == "R":
            side_cubies_old = np.array([1, 3, 7, 15, 21, 23, 18, 10])
            face_cubies_old = np.array([[8, 9], [16, 17]])
        elif move_side == "L":
            side_cubies_old = np.array([2, 0, 11, 19, 22, 20, 14, 6])
            face_cubies_old = np.array([[4, 5], [12, 13]])
        elif move_side == "F":
            side_cubies_old = np.array([2, 3, 13, 5, 21, 20, 8, 16])
            face_cubies_old = np.array([[6, 7], [14, 15]])
        elif move_side == "B":
            side_cubies_old = np.array([0, 1, 9, 17, 23, 22, 12, 4])
            face_cubies_old = np.array([[10, 11], [18, 19]])
        elif move_side == "U":
            side_cubies_old = np.array([6, 7, 8, 9, 10, 11, 4, 5])
            face_cubies_old = np.array([[0, 1], [2, 3]])
        elif move_side == "D":
            side_cubies_old = np.array([14, 15, 12, 13, 18, 19, 16, 17])
            face_cubies_old = np.array([[20, 21], [22, 23]])

        side_cubies_new = np.roll(side_cubies_old, -2 * repetitions)
        face_cubies_new = np.rot90(face_cubies_old, 4 - repetitions).flatten()
        face_cubies_old = face_cubies_old.flatten()

        np.put(self.cube, side_cubies_old, self.cube[side_cubies_new])
        np.put(self.cube, face_cubies_old, self.cube[face_cubies_new])

    def algorithm(self, moves):
        for move in moves.split(" "):
            if len(move) == 2:
                self.move(move[0], move[1])
            else:
                self.move(move[0])

    def step(self, action):
        done = False
        termintated = False
        
        
        # if action == self.action_last:
        #     self.s_continue += 1
        
        # else:
        #     self.s_continue = 0
        
        # if self.s_continue == 3:
        #     # The previous successful version
        #     # return None, -5000, False, {}, True, {} 
        #     return None, -100, False, {}, True, {}

        # self.action_last = action
        

        move = ACTION_MAP[action]
        self.move(move[0], move[1])
        self.scramble_this_time += " {}".format(ACTION_MAP[action][0])
        self.update_cube_reduced()
        self.update_cube_state()

        self.move_len += 1
        observation = deepcopy(self.cube)
        self.now_cube = deepcopy(self.cube)
        
        reward, done = self.reward()
        
        if self.move_len >= 100:
            done = True
            # The previous successful version
            # reward = -100
            reward = -100

        if reward == 100:
            done = True
            termintated = True



        info = {"cube": deepcopy(self.cube), "cube_reduced": self.cube_reduced}

        return observation, reward, termintated, {}, done, info

    # def reward(self):
    #     if self.cube_reduced == "WWWWOOGGRRBBOOGGRRBBYYYY":
    #         return 100, True
    #     else:
    #         return -1, False
    def reward(self):
        is_done = False

        if self.if_nx_module:
            distance = get_Dist(self.ori_cube, self.now_cube, self.nx_module.nx_network, self.nx_module.device)
        
            if self.now_cube.tolist() == self.ori_cube.tolist():
                is_done = True
                # The previous successful version
                # reward = 100
                reward = 100
                print("Congratulations!!!!")
            
            else:
                try: 
                    reward = - math.log(distance, 1.2)
                
                except ValueError:
                    reward = -math.log(distance + 1e-4, 1.2)
        
        else:
            if self.now_cube.tolist() == self.ori_cube.tolist():
                is_done = True
                reward = 100
                print("Congratulations!!!!")

            else:
                reward = -1
                pass

        return reward, is_done

    def reset(self, scramble=None):
        self.cube = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                             dtype=np.uint8)
        if scramble:

            self.scramble_this_time = scramble
            self.algorithm(scramble)

        elif scramble == False:
            self.scramble_this_time = ""
            pass

        else:
            self.scramble_this_time = self.generate_scramble()
            # print(self.scramble_this_time)
            self.algorithm(self.scramble_this_time)

        self.now_cube = deepcopy(self.cube)
        
        self.update_cube_reduced()
        self.update_cube_state()

        self.s_continue = 0
        self.action_last = None

        self.move_len = 0

        return deepcopy(self.cube), {}

    def render(self, mode='human', render_time=100):
        if mode == 'ansi':
            return self.cube_reduced
        else:
            cube_to_render = [[0] * 2 + [1] * 2 + [0] * 4] * 2 + [[1] * 8] * 2 + [[0] * 2 + [1] * 2 + [0] * 4] * 2
            render_array = np.zeros((6, 8, 3), dtype=np.uint8)
            ctr = 0
            for row in range(6):
                for col in range(8):
                    if cube_to_render[row][col] == 1:
                        render_array[row][col] = COLOR_MAP[TILE_MAP[self.cube[ctr]]]
                        ctr += 1
            if mode == 'rgb_array':
                return render_array
            elif mode == "human":
                img = cv2.cvtColor(render_array, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (300, 225), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Cube", np.array(img))
                cv2.waitKey(render_time)

    def close(self):
        del self.cube_states
        gc.collect()
        cv2.destroyAllWindows()



TILE_MAP = {
    0: 'W', 1: 'W', 2: 'W', 3: 'W',
    4: 'O', 5: 'O', 6: 'G', 7: 'G', 8: 'R', 9: 'R', 10: 'B', 11: 'B',
    12: 'O', 13: 'O', 14: 'G', 15: 'G', 16: 'R', 17: 'R', 18: 'B', 19: 'B',
    20: 'Y', 21: 'Y', 22: 'Y', 23: 'Y'
}

COLOR_MAP = {
    'W': (255, 255, 255),
    'O': (255, 165, 0),
    'G': (0, 128, 0),
    'R': (255, 0, 0),
    'B': (0, 0, 255),
    'Y': (255, 255, 0)
}

ACTION_MAP = {0: ("F", None), 1: ("R", None), 2: ("U", None), 
              3: ("F", "'"), 4: ("R", "'"), 5: ("U", "'")
              }


if __name__ == "__main__":
    env = RubiksCube222Env()
    env.reset()

