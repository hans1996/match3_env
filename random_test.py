from gym_match3.envs import Match3Env
import gym
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import torch
import cv2
from PIL import Image
from gym_match3.envs.levels import LEVELS #  default levels
from gym_match3.envs.levels import Match3Levels, Level
from gym_match3.envs.game import (Board,
                                  RandomBoard,
                                  CustomBoard,
                                  Point,
                                  Cell,
                                  AbstractSearcher,
                                  MatchesSearcher)
from gym_match3.envs.game import OutOfBoardError, ImmovableShapeError
from random import choice
from configparser import ConfigParser, ExtendedInterpolation





parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read('configure.ini')

width_hight = int(parser.get('gym_environment','board_width_and_hight')) 
n_shapesss = int(parser.get('gym_environment','board_number_of_different_color'))

def Getlevels(WnH,shapes):
    LEVELS = [Level(WnH,WnH,shapes, np.zeros((WnH,WnH)).tolist())]
    return LEVELS

env = Match3Env(levels=Match3Levels(Getlevels(width_hight,n_shapesss)))

class One_hot(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        
    def observation(self, observation):
        observation = observation.reshape(width_hight ,width_hight).astype(int)
        grid_onehot = np.zeros(shape=(env.n_shapes+1, width_hight , width_hight ))
        table = {i:i for i in range(-1,n_shapesss)} 
        
        for i in range(width_hight):
            for j in range(width_hight):
                grid_element = observation[i][j]
                grid_onehot[table[grid_element]][i][j]=1
        print(grid_onehot)        
        return grid_onehot

#env = One_hot(env)

available_actions = {v : k for k, v in dict(enumerate(env.get_available_actions())).items()}

for i_episode in range(1): #玩 1次遊戲
     
    observation = env.reset() 
    total_reward = 0
    q  = 0
    while True:
        env.render()
       
        validate_move = env.possible_move   # 一般用 env 的屬性紀錄合法走步
        
        #一開始遊戲初始化時,env的屬性會是 None,以及重新一場遊戲時要再 get一次合法走步
        if validate_move == None or len(validate_move)== 0:  
      
            validate_move = env.get_validate_actions()
        
        #print(validate_move)
        
        validate_list = []       
        for i in validate_move:
            if i in available_actions:
                validate_list.append(available_actions.get(i))   
         

        action = choice(validate_list)
        
        
        observation, reward, done, info = env.step(action)  # step函數會檢查下一個 observation 有沒有合法走步,並回傳 observation,如果沒有合法走步 done == True
        q  = q + 1    
        total_reward = total_reward + reward
        print()
        print('observation:')  
        print(observation)
        print('total_reward: ',total_reward)
        print('step',q)
        #print('the swap of coordinate is: ',list(env.get_available_actions()[action]))

        if done:           
            break