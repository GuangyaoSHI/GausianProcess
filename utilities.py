# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:42:15 2021

@author: sgyhi
"""
import networkx as nx
import matplotlib.pyplot as plt
from gamestate_no_attack import *
from policies_no_attack import *
import random
import copy
from itertools import combinations
from itertools import product
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
# https://graphviz.readthedocs.io/en/latest/api.html#graphviz.Source.view
from graphviz import Source


#
# def setup_game():
#     # define a blank board
#     # nx.draw(G, with_labels=True)
#     grid_len = 5
#     grid_height = 5
#     G = nx.grid_2d_graph(grid_len, grid_height)
#     values = [1, 2, 23, 4, 5,
#               2, 3, 4, 2, 7,
#               3, 6, 27, 8, 1,
#               5, 6, 27, 2, 9,
#               12, 1, 23, 4, 9]
#     reward = dict(zip(list(G.nodes), values))
#     nx.set_node_attributes(G, reward, 'reward')
#     return G


def setup_game():
    # define a blank board
    G = nx.Graph()
    # {node:reward}
    nodes = {(-4, 0): 0, (-3, 0): 0, (-2, 0): 0, (-2, 1): 0,
             (-2, 2): 0, (-1, 0): 0, (0, 0): 0,
             (1, 0): 1, (2, 0): 1, (2, 1): 2, (2, 2): 20, (3, 0): 2, (4, 0): 10}
    for node in nodes:
        G.add_node(node, reward=nodes[node], position=node)
    edges = [((-4, 0), (-3, 0)), ((-3, 0), (-2, 0)), ((-2, 0), (-2, 1)), ((-2, 1), (-2, 2)), ((-2, 0), (-1, 0)),
             # ((-1,0),(0,0)),\
             ((1, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 1), (2, 2)), ((2, 0), (3, 0)), ((3, 0), (4, 0)), ((0, 0), (1, 0))]
    G.add_edges_from(edges)
    return G


# pos = nx.get_node_attributes(G,'position')
# labels = nx.get_node_attributes(G,'reward')
# nx.draw(G, pos=pos, labels=labels, with_labels=True)

def plot_reward_map(gamestate, ax):
    pos = {}
    labels = {}
    for node in gamestate.G.nodes:
        pos[node] = node
        labels[node] = gamestate.G.nodes[node]['reward']
    nx.draw(gamestate.G, pos=pos, ax=ax, labels=labels, with_labels=True)


def plot_path(path, ax, style):
    X = []
    Y = []
    for node in path:
        X.append(node[0])
        Y.append(node[1])
    # style = random.sample(['ro-','b*-','go-','mo-','yo-','ko-'],1)[0]
    ax.plot(X, Y, style)

def path_animation(gameState, paths):
    # https://eli.thegreenplace.net/2016/drawing-animated-gifs-with-matplotlib/
    
    return

def all_paths(starts, game):
    trees = {}
    curr_nodes = {}
    node_counter = {}
    for robot in starts:
        node_counter[robot] = 0
        trees[robot] = nx.DiGraph()
        # keep track of the nodes at current depth
        # they are positions on the board
        curr_nodes[robot] = {node_counter[robot]: starts[robot][0]}
        trees[robot].add_node(node_counter[robot], pos=starts[robot][0])
        node_counter[robot] += 1

    for i in range(game.horizon):
        next_nodes = {}
        for robot in starts:
            # print('expand robot {}'.format(robot))
            next_nodes[robot] = {}
            # print('current frontier: {}'.format(curr_nodes[robot]))
            for node in curr_nodes[robot]:
                # print('consider node {}'.format(node))
                position = curr_nodes[robot][node]
                # print('its position is {}'.format(position))
                # print('its neighbor are {}'.format(list(game.G.neighbors(position))))
                for neighbor in game.G.neighbors(position):
                    # print('{} is added'.format(neighbor))
                    next_nodes[robot][node_counter[robot]] = neighbor
                    trees[robot].add_node(node_counter[robot], pos=neighbor)
                    trees[robot].add_edge(node, node_counter[robot])
                    node_counter[robot] += 1
        curr_nodes = copy.deepcopy(next_nodes)

        # pos = nx.get_node_attributes(trees[0], 'pos')
    # nx.draw(trees[0],  with_labels=True)
    paths = {}
    for r in starts:
        paths[r] = []
        for node in trees[r]:
            if trees[r].out_degree(node) == 0:
                sp = nx.shortest_path(trees[r], 0, node)
                path = [trees[r].nodes[node]['pos'] for node in sp]
                # if len(list(set(path))) == len(path):
                #     # print(path)
                paths[r].append(path)
    return paths


def all_attacks(robots, game):
    '''

    Parameters
    ----------
    robots : list [0,1,2]
        DESCRIPTION.
    game : game state
        DESCRIPTION.

    Returns
    -------
    attacks : all posstible attacks to robots at any time
        DESCRIPTION.

    '''
    # return all attacks [{(robots):(time)}]
    H = game.horizon
    alpha = game.ALPHA
    attacks = []
    for pair in combinations(robots, alpha):
        # print('this time attack robots {}'.format(pair))
        L = len(pair)
        T = [list(range(0, H + 1)) for i in range(L)]
        for att_time in product(*T):
            # print('attack time is {}'.format(att_time))
            attacks.append(dict(zip(pair, att_time)))
    return attacks


def compute_reward(game, path, attack):
    '''
    Parameters
    ----------
    game : game state
        DESCRIPTION.
    path : (path1, path2,) each path is a list [(0,1),(2,3)]
        DESCRIPTION.
    attacks : {robot:attack time} 
        DESCRIPTION.

    Returns
    -------
    reward

    '''
    visited_nodes = []
    for r in range(len(path)):
        if r in attack.keys():
            visited_nodes += path[r][0:attack[r] + 1]
        else:
            visited_nodes += path[r]

    visited_nodes = list(set(visited_nodes))
    reward = 0
    for node in visited_nodes:
        reward += game.G.nodes[node]['reward']
    return reward


def action_selection(MCTS):
    # MCTS is monte carlo search tree

    # {reward:action}

    for node in MCTS.digraph.successors(0):
        reward = MCTS.digraph.nodes[node]['reward'] / MCTS.digraph.nodes[node]['n']
        # reward is a np.array
        # define your rule here to select a node
    next_node = {}
    return next_node


def visualize_MCTS(MCTS, fileName='MCTS.dot'):
    # each node it has information on robots' position
    # reward, n, and UCT
    mapping = {}
    for node in MCTS.digraph.nodes:
        state = MCTS.digraph.nodes[node]['state'].currNode
        reward = MCTS.digraph.nodes[node]['reward']
        n = MCTS.digraph.nodes[node]['n']
        uct = MCTS.digraph.nodes[node]['uct']
        mapping[node] = '-Node:' + str(node) + \
                        ' r:' + str(reward) + \
                        ' n:' + str(n) + \
                        ' uct:' + str(uct) + '\n' + \
                        's:' + str(state)
    G = copy.deepcopy(MCTS.digraph)
    G = nx.relabel_nodes(G, mapping)
    write_dot(G, fileName)
    s = Source.from_file(fileName)
    # s.view()
    return s
