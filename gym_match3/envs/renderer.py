import numpy as np

from gym_match3.envs.game import Board
from gym_match3.envs.game import Game, Point

from PIL import Image      
import cv2

class Renderer():
    def __init__(self, h, w, n_shapes):

        self.__n_shapes = n_shapes
        self.__height = h
        self.__weight = w
        self.d = {  0: (255, 0, 0),   
                    1: (0, 255, 255),      
                    2: (0, 0, 255) ,      
                    3: (230,224,176) ,    
                    4: (0,255,0),             
                    5: (100,100,100),
                    6:(150,150,150),
                    7:(200,200,200),
                    8:(200,150,250),
                    9:(250,200,150),
                    10:(150,200,250) }    

    
    def render_board(self, board: Board.board):      
        background = np.zeros((self.__height,self.__weight,3), dtype=np.uint8)
        
        for color in range(self.__n_shapes):
            result = np.where(board == color)
            listOfCoordinates= list(zip(result[0], result[1]))            
            for cord in listOfCoordinates:
                background[cord] = self.d[color]

        img = Image.fromarray(background, 'RGB')      
        cv2.imshow("image", np.array(img))
        cv2.waitKey(200)    

 