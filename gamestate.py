# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:04:42 2020

@author: sgyhi
"""

"""
Multi-robot Orienteering Problem as a game
<state representation, transition function, move function, \
    terminal detector>
"""
import networkx as nx
import copy
from itertools import combinations
import numpy as np


class GameState:
    def __init__(self, starts, horizon):
        # starts is a dictionary of starting positions of robots
        # starts = {robot: position (x, y)}
        self.horizon = horizon
        # trajectories/paths of robots and attackers
        # {robot:[t0,t1,...]}
        self.paths_robots = {}
        for robot in starts:
            self.paths_robots[robot] = [starts[robot]]
        self.currNode = starts

    def move(self, next_node):
        # next_node is a dictionary
        # next_node = {robot:position}
        for robot in self.paths_robots:
            self.paths_robots[robot].append(next_node[robot])
        self.currNode = next_node
        self.horizon -= 1

    def legal_moves(self):
        """
        currNode = {robot:position}
        return the neighbors of the current node
        """
        possible_moves = {}
        # example
        for robot in self.currNode:
            x, y = self.currNode[robot]
            possible_moves[robot] = [(x-1, y), (x+1, y), (x, y+1), (x, y-1)]
        return possible_moves

    def transition_function(self, next_node):
        # verify that the next position is legal
        # next_node = {robot:position}
        for robot in next_node:
            assert next_node[robot] in self.legal_moves(self.currNode)[robot]
        # First, make a copy of the current state
        new_state = copy.deepcopy(self)

        # Then, apply the action to produce the new state
        new_state.move(next_node)

        return new_state

    def is_terminal(self):
        if self.horizon > 0:
            return False
        else:
            return True

    def collected_reward(self):
        # compute the reward associated with the current state
        total = np.array([0, 0])
        for robot in self.paths_robots:
            # GP(self.paths_robots[robot])
            total += np.array([1, 0])
        return total


