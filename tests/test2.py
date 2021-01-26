{
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6-final"
  },
  "orig_nbformat": 2,
  "kernelspec": {
   "name": "python3",
   "display_name": "Python 3",
   "language": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2,
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import unittest\n",
    "import numpy as np\n",
    "from gym_match3.envs.game import (Board,\n",
    "                                  RandomBoard,\n",
    "                                  CustomBoard,\n",
    "                                  Point,\n",
    "                                  Cell,\n",
    "                                  AbstractSearcher,\n",
    "                                  MatchesSearcher,\n",
    "                                  Filler,\n",
    "                                  Game,\n",
    "                                  MovesSearcher,\n",
    "                                  OutOfBoardError,\n",
    "                                  ImmovableShapeError)\n",
    "from gym_match3.envs.levels import (Match3Levels,\n",
    "                                    Level)\n",
    "\n",
    "\n",
    "class TestBoard(unittest.TestCase):\n",
    "    def setUp(self):\n",
    "        self.board = Board(columns=2, rows=2, n_shapes=3)\n",
    "        board = np.array([\n",
    "            [0, 1],\n",
    "            [2, 0]\n",
    "        ])\n",
    "        self.board.set_board(board)"
   ]
  }
 ]
}