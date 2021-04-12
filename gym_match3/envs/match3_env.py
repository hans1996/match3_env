import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_match3.envs.game import Game, Point
from gym_match3.envs.game import OutOfBoardError, ImmovableShapeError
from gym_match3.envs.levels import LEVELS, Match3Levels
from gym_match3.envs.renderer import Renderer

from gym_match3.envs.game import MovesSearcher, MatchesSearcher, Filler


from itertools import product
import warnings
import numpy as np
from pathlib import Path

from configparser import ConfigParser, ExtendedInterpolation

import os


path_current_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
path_config_file = os.path.join(path_current_directory,'configure.ini')
config = ConfigParser()
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read(path_config_file)


BOARD_NDIM = 2


class Match3Env(gym.Env):
    
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, rollout_len=100, all_moves=True, levels=None, random_state=None):

        self.rollout_len = rollout_len
        self.random_state = random_state
        self.all_moves = all_moves
        self.levels = levels or Match3Levels(LEVELS)
        self.h = self.levels.h
        self.w = self.levels.w
        self.n_shapes = self.levels.n_shapes
        self.episode_counter = 0
        self.possible_move = random_state
        self.game = Game(
            rows=self.h,
            columns=self.w,
            n_shapes=self.n_shapes,
            length=3,
            all_moves=all_moves,
            random_state=self.random_state,
            
            )
        self.reset()[np.newaxis,:]
        self.renderer = Renderer(self.levels.h, self.levels.w, self.n_shapes)

        self.step_add_immovable = parser.getboolean('gym_environment','step_add_immovable')
        self.number_step_add_immovable = int(parser.get('gym_environment','number_of_step_add_immovable'))
        self.number_of_step_immovable_add = int(parser.get('gym_environment','number_of_step_immovable_add')) 
        
        self.match_counts_add_immovable = parser.getboolean('gym_environment','match_counts_add_immovable')
        self.number_match_counts_add_immovable = int(parser.get('gym_environment','number_of_match_counts_add_immovable'))
        self.number_of_match_counts_immovable_add  = int(parser.get('gym_environment','number_of_match_counts_immovable_add'))

        # setting observation space
        self.observation_space = spaces.Box(
            low=0,
            high=self.n_shapes,
            shape=(1,self.h,self.w),
            dtype=int)

        # setting actions space
        self.__match3_actions = self.get_available_actions()
        self.action_space = spaces.Discrete(
            len(self.__match3_actions))

    @staticmethod
    def get_directions(board_ndim):
        """ get available directions for any number of dimensions """
        directions = [
            [[0 for _ in range(board_ndim)] for _ in range(2)]
            for _ in range(board_ndim)
        ]
        for ind in range(board_ndim):
            directions[ind][0][ind] = 1
            directions[ind][1][ind] = -1
        return directions

    def points_generator(self):
        """ iterates over points on the board """
        rows, cols = self.game.board.board_size
        points = [Point(i, j) for i, j in product(range(rows), range(cols))]
        for point in points:
            yield point

    def get_available_actions(self):
        """ calculate available actions for current board sizes """    
        actions = []  
        direction = [[1, 0], [0, 1]]
        for dir_ in direction:
            for point in self.points_generator():                
                dir_p = Point(*dir_)
                new_point = point + dir_p
                try:
                    _ = self.game.board[new_point]                    
                    actions.append((point, new_point))
                except OutOfBoardError:
                    continue
        return actions 

    def get_validate_actions(self):
        possible = self.game.get_possible_moves()
        validate_actions = []
        for point, direction in possible:
            newpoint =  point +  Point(*direction)
            validate_actions.append((newpoint, point))     
        return list(validate_actions)


    def get_action(self, ind):
        return self.__match3_actions[ind]


    def reset(self, *args, **kwargs):
        board = self.levels.sample()
        self.game.start(board)
        return self.get_board()[np.newaxis,:]

    def swap(self, point1, point2):
        try:
            reward = self.game.swap(point1, point2)
        except ImmovableShapeError:
            reward = 0
        return reward

    def get_board(self):
        return self.game.board.board


    def render(self, mode='human'):
        if mode == 'human':
            return self.renderer.render_board(self.game.board.board) 
        else:
            super(Match3Env, self).render(mode=mode) # just raise an exception


    def step(self, action):

        self.episode_counter += 1
        m3_action = self.get_action(action)
        reward = self.swap(*m3_action)
        ob = self.get_board()[np.newaxis,:]
        self.possible_move = self.get_validate_actions()

        if self.step_add_immovable:

            if self.episode_counter % self.number_step_add_immovable == 0:
                self.generate_immovable(self.number_of_step_immovable_add)
    
        if self.match_counts_add_immovable:
            if self.game.matchs_counter > self.number_match_counts_add_immovable:                
                self.generate_immovable(self.number_of_match_counts_immovable_add)
                self.game.matchs_counter = 0

        if len(self.possible_move ) == 0:
            episode_over = True
            self.episode_counter = 0
        else:
            episode_over = False
        

        return ob, reward, episode_over, {}



    def generate_immovable(self,number_of_immovable):
        obs = self.get_board()
        A = np.random.randint(obs.shape, size=(number_of_immovable,2))
        for i in range(number_of_immovable):
            if obs[A[i][0],A[i][1]] == -1:
                self.generate_immovable(1)
            else:
                obs[A[i][0],A[i][1]] = -1




    
    



