import unittest
import numpy as np
from gym_match3.envs.game import (Board,
                                  RandomBoard,
                                  CustomBoard,
                                  Point,
                                  Cell,
                                  AbstractSearcher,
                                  MatchesSearcher,
                                  Filler,
                                  Game,
                                  MovesSearcher,
                                  OutOfBoardError,
                                  ImmovableShapeError)
from gym_match3.envs.levels import (Match3Levels,
                                    Level)


def setUp():
    board = Board(columns=2, rows=2, n_shapes=3)
    board1 = np.array([
        [0, 1],
        [2, 0]
    ])
    board.set_board(board1)
    #print(board.board)
    


#print(setUp())




correct = np.array([[1, 0], [2, 0]])
#print(correct)

board = Board(columns=2, rows=2, n_shapes=3)
board1 = np.array([
    [0, 1],
    [2, 0]
])
board.set_board(board1)

board.get_shape