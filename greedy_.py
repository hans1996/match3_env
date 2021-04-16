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
import copy


parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read('configure.ini')


def Getlevels(WnH,shapes):
    LEVELS = [Level(WnH,WnH,shapes, np.zeros((WnH,WnH)).tolist())]
    return LEVELS

env = Match3Env(
    step_add_immovable = parser.getboolean('gym_environment','step_add_immovable'),
    number_of_step_add_immovable = int(parser.get('gym_environment','number_of_step_add_immovable')),
    match_counts_add_immovable = parser.getboolean('gym_environment','match_counts_add_immovable'),
    number_of_match_counts_add_immovable = int(parser.get('gym_environment','number_of_match_counts_add_immovable')), 
    train_or_test = parser.get('gym_environment','train_or_test'),
    rollout_len = int(parser.get('gym_environment','rollout_len')),
    levels=Match3Levels(Getlevels(int(parser.get('gym_environment','board_width_and_hight')),
    int(parser.get('gym_environment','board_number_of_different_color')))),
    immovable_move_ = parser.getboolean('gym_environment','immovable_move'),
    n_of_match_counts_immov = int(parser.get('gym_environment','number_of_immovable_add')),
    no_legal_shuffle_or_new_ = parser.get('gym_environment','no_legal_shuffle_or_new'))



available_actions = {v : k for k, v in dict(enumerate(env.get_available_actions())).items()}


reward_list = []
for i_episode in range(10): #玩 1次遊戲
    env.reset()
    
    env.game.greedy_actions = False    
    total_reward = 0
    
    while True:
        observation_orginal = copy.deepcopy(env) 
        
        env.render()

        validate_move = env.possible_move   # 一般用 env 的屬性紀錄合法走步

        #一開始遊戲初始化時,env的屬性會是 None,以及重新一場遊戲時要再 get一次合法走步
        if validate_move == None or len(validate_move)== 0:   
            validate_move = env.get_validate_actions() 

        validate_list = []       
        for i in validate_move:
            if i in available_actions:
                validate_list.append(available_actions.get(i))   
                 


        temp_reward_dict = {}
        for action in validate_list:
            env.game.greedy_actions = True
            observation, reward, done, info = env.step(action)
            temp_reward_dict[action] = reward
            env = copy.deepcopy(observation_orginal) 
        #print('env.game.matchs_counter:',env.game.matchs_counter)
        print(temp_reward_dict)
        
        print('env.game.matchs_counter:',env.game.matchs_counter)    
        max_key = max(temp_reward_dict, key=temp_reward_dict.get)
        print('max_key:',max_key)
        observation, reward, done, info = env.step(action = max_key)
        
        print('observation:',observation)
        #print('i neeed:' , env.number_of_match_counts_add_immovable)


      
        total_reward = total_reward + reward
        


        print('reward: ', reward)
        #print('env.game.current_move_reward: ',env.game.current_move_reward)
        #print('the swap of coordinate is: ',list(env.get_available_actions()[action]))

        if done: 
            print(i_episode)          
            break
    reward_list.append(total_reward)
