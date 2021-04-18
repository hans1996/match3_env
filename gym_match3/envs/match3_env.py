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


#path_current_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
#path_config_file = os.path.join(path_current_directory,'configure.ini')
#config = ConfigParser()
#parser = ConfigParser(interpolation=ExtendedInterpolation())
#parser.read(path_config_file)


BOARD_NDIM = 2


class Match3Env(gym.Env):
    
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,step_add_immovable, 
                 number_of_step_add_immovable,
                 match_counts_add_immovable,
                 number_of_match_counts_add_immovable,
                 immovable_move_, 
                 n_of_match_counts_immov,
                 train_or_test,
                 no_legal_shuffle_or_new_,
                 rollout_len=100, all_moves=True, 
                 levels=None, random_state=None):

        self.step_add_immovable = step_add_immovable
        self.number_of_step_add_immovable = number_of_step_add_immovable
        self.match_counts_add_immovable = match_counts_add_immovable
        self.number_of_match_counts_add_immovable = number_of_match_counts_add_immovable
        self.train_or_test = train_or_test
        self.rollout_len = rollout_len
        self.immovable_move = immovable_move_
        self.n_of_match_counts_immov = n_of_match_counts_immov
        self.no_legal_shuffle_or_new = no_legal_shuffle_or_new_       
        

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
            no_legal_shuffle_or_new=self.no_legal_shuffle_or_new,
            number_of_match_counts_add_immovable = self.number_of_match_counts_add_immovable,
            train_or_test= self.train_or_test,
            filler=Filler(immovable_move=self.immovable_move,
            n_of_match_counts_immov =self.n_of_match_counts_immov,
            number_of_match_counts_add_immovable = self.number_of_match_counts_add_immovable,
            match_counts_add_immovable = self.match_counts_add_immovable)
            ) 
        
        self.reset()[np.newaxis,:]
        self.renderer = Renderer(self.levels.h, self.levels.w, self.n_shapes)

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
        return self.game.board.board.copy()


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
        
        self.game.filler.immovable = False

        #self.match_counts_immovable()
        self.step_immovable()


        if self.train_or_test == 'train':
            if len(self.possible_move ) == 0:
                episode_over = True
                self.episode_counter = 0
                self.game.filler.immovable = False
            else:
                episode_over = False
            return ob, reward, episode_over, {}

        elif self.train_or_test == 'test':
            if self.episode_counter >= self.rollout_len:
                episode_over = True
                self.episode_counter = 0
                self.game.filler.immovable = False        
            else:
                episode_over = False

            return ob, reward, episode_over, {}


    def match_counts_immovable(self):
        if self.match_counts_add_immovable:            
            if self.game.matchs_counter > self.number_of_match_counts_add_immovable:                
                #self.generate_immovable(self.number_of_match_counts_immovable_add, Fall = parser.getboolean('gym_environment','immovable_Fall'),temp_counter = self.h ,temp_counter2 = 0)
                self.game.filler.immovable = True          
                self.game.matchs_counter = 0
            else:
                self.game.filler.immovable = False


    def step_immovable(self):
        if self.step_add_immovable:
            if self.episode_counter % self.number_of_step_add_immovable == 0:
                self.game.filler.immovable = True
            else:
                self.game.filler.immovable = False     



'''


    def generate_immovable(self, number_of_immovable, Fall , temp_counter ,temp_counter2 ):
        obs = self.get_board()
               
        if not Fall:
            A = np.random.randint(obs.shape, size=(number_of_immovable,2))    
            for i in range(number_of_immovable):
                if obs[A[i][0],A[i][1]] == -1:
                    self.generate_immovable(1, Fall, temp_counter ,temp_counter2 )
                else:
                    obs[A[i][0],A[i][1]] = -1                      
        else:
            A = np.random.randint(obs.shape[0],size=number_of_immovable)
            temp_counter = temp_counter
            temp_counter2 = temp_counter2

            if not np.all(obs[temp_counter-1] == -1):
                for i in range(number_of_immovable):
                    if obs[temp_counter-1,A[i]] == -1 :
                        self.generate_immovable(1,Fall,temp_counter,temp_counter2)
                    else:
                        obs[temp_counter-1,A[i]] = -1 
            else:               
                self.generate_immovable(number_of_immovable,Fall,temp_counter-1,temp_counter2+1)


    def step_immovable(self):
        if self.step_add_immovable:
            if self.episode_counter % self.number_step_add_immovable == 0:
                self.generate_immovable(self.number_of_step_immovable_add, Fall = parser.getboolean('gym_environment','immovable_Fall'),temp_counter = self.h ,temp_counter2 = 0 )

'''              