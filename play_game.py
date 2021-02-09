# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:36:52 2020

@author: sgyhit
"""

from gamestate import *
from policies import *
import networkx as nx
import sys
import copy
from utilities import *
import matplotlib.pyplot as plt
import random
import pickle


def mcts_process(game, budget=400):
    '''
    game is gameState object
    budget is computational budget for Monte Carlo Tree Search
    '''
    MCTS = MCTSPolicy(game)
    for i in range(budget):
        # select the node to expand in search tree
        node_exp = MCTS.selection(0)
        # expand the node and return the frontier node
        node_fron = MCTS.expansion(node_exp)
        # rollout from frontier node
        reward = MCTS.simulation(node_fron)
        # backpropagation
        MCTS.backpropagation(node_fron, reward)
        # make a copy of the current tree
    # s = visualize_MCTS(MCTS)
    # s.view()
    # you need to rewrite this action_selection function
    nextNode = action_selection(MCTS)
    return (nextNode, MCTS)

def terminal_condition():
    return True

def play_game(starts, budget):

    # start a game
    horizon = 10
    game = GameState(starts, horizon)

    # computational budget
    path = [starts]
    while terminal_condition():
        next_node, mcts = mcts_process(game, budget)
        path.append(next_node)
        game = GameState(next_node, horizon)

    trajs = {}
    for robot in path[0]:
        traj = [node[robot] for node in path]
        print('robot {} trajectory:'.format(robot))
        print(traj)
        trajs[robot] = traj

    print('reward is {}'.format(game.collected_reward()))

    return game.collected_reward()


if __name__ == "__main__":
    # budgets = [100, 400, 800, 1600]
    starts = {0: (0, 0), 1: (0, 0), 2: (0, 0), 3: (0, 0)}
    play_game(starts, 800)
