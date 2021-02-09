# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:00:49 2020

@author: sgyhi
"""

from abc import ABCMeta, abstractmethod
import random
import numpy as np
import operator
import networkx as nx
import copy
from gamestate import GameState
import sys
import random
from itertools import product


class Policy(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def move(self, state):
        pass


class RandomPolicy(Policy):
    def move(self, state):
        """
        Chooses moves from the legal moves in a given state
        return next_node = {robot:[position, attack_indicator]}   
        """
        # print(state.legal_moves(state.currNode))
        if not state.legal_moves(state.currNode):
            # there are no available actions
            return {}
        if state.turn == 'robot':
            # if it is robots' turn legal_moves = {robot:[actions]}
            assert type(state.legal_moves(state.currNode)) == dict
            next_node = copy.deepcopy(state.currNode)
            for robot in next_node:
                next_node[robot][0] = random.sample(state.legal_moves(state.currNode)[robot], 1)[0]
            return next_node
        else:
            # if it is attacker's turn legal_moves = [{robot:attack_indicator}]
            assert type(state.legal_moves(state.currNode)) == list
            next_node = copy.deepcopy(state.currNode)
            attacker_move = random.sample(state.legal_moves(state.currNode), 1)[0]
            for robot in next_node:
                next_node[robot][1] = attacker_move[robot]
            return next_node


class MCTSPolicy(Policy):
    """
    Implementation of Monte Carlo Tree Search
    
    """

    def __init__(self, root_state):
        # robot_state = {robot:[pos, attack_indicator]}
        self.digraph = nx.DiGraph()
        self.EPSILON = 10e-6  # Prevents division by 0 in calculation of UCT

        # Constant parameter to weight exploration vs. exploitation for UCT
        # check other implementations for reward instead of win/lose case
        # https://github.com/GuangyaoSHI/mcts/blob/master/include/mcts/defaults.hpp
        self.uct_c = 2 * np.sqrt(2)
        self.node_counter = 0
        self.digraph.add_node(self.node_counter,
                              reward=0,
                              n=0,
                              uct=sys.maxsize,
                              state=root_state)
        self.node_counter += 1
        # self.who_is_play = root_state.turn

    def is_leaf_node(self, node):
        if (self.digraph.in_degree(node) == 0) \
                and (self.digraph.out_degree(node) == 0):
            return True
        elif (self.digraph.in_degree(node) == 1) \
                and (self.digraph.out_degree(node) == 0):
            return True
        else:
            return False

    def selection(self, root):
        """
        Starting at root, recursively select the best node that maximizes UCT
        until a node is reached that has no explored children
        Keeps track of the path traversed by adding each node to path as
        it is visited
        :return: the node to expand
        """
        # print('we are selecting')

        if self.is_leaf_node(root):
            return root
        else:
            # the current node is not a leaf node
            # print('root is not a leaf node')
            # handle the general case
            children = self.digraph.successors(root)
            uct_values = {}
            for child_node in children:
                uct_values[child_node] = self.uct(node=child_node, parent=root)

            # Choose the child node that maximizes the expected value given by UCT
            best_child_node = max(uct_values.items(), key=operator.itemgetter(1))[0]

            return self.selection(best_child_node)

    def expansion(self, node):
        # if n == 0, just rollout
        # if n >0, expand all children and select one to rollout
        if self.digraph.nodes[node]['n'] == 0:
            return node
        else:
            curr_game_pos = self.digraph.nodes[node]['state'].currNode
            # robot turn: legal_moves = {robot:[actions]}
            # attacker turn: legal_moves= [{robot:indicator}]
            legal_moves = self.digraph.nodes[node]['state'].legal_moves(curr_game_pos)
            # print('Legal moves: {}'.format(legal_moves))
            # if node is a terminal state, e.g. run out of budget
            # legal_moves will be empty
            if not legal_moves:
                return node
        # for each available action from the current state, add all new states
        # to the tree
        child_node_id = []
        if self.digraph.nodes[node]['state'].turn == 'robot':
            # print('We are expanding robots actions')
            # legal_moves = {robot:[actions]}
            # Todo: if some robots have empty set, the following code
            # will return empty set on joint_moves
            # !!!!!this is a problem!!!!
            joint_actions = list(product(*list(legal_moves.values())))
            # check whether the joint action set is empty
            assert joint_actions, 'joint action set is empty'
            joint_moves = [dict(zip(list(legal_moves.keys()), joint_action)) \
                           for joint_action in joint_actions]
            for move in joint_moves:
                # from move = {robot:position}
                # to node = {robot:[position, indicator]}
                next_node = copy.deepcopy(curr_game_pos)
                for robot in next_node:
                    next_node[robot][0] = move[robot]
                    # print('adding to expansion analysis with: {}'.format(next_node))
                child = self.digraph.nodes[node]['state'].transition_function(next_node)
                self.digraph.add_node(self.node_counter,
                                      reward=0,
                                      n=0,
                                      uct=sys.maxsize,
                                      state=child)
                self.digraph.add_edge(node, self.node_counter, action=next_node)
                child_node_id.append(self.node_counter)
                self.node_counter += 1
            # return first new child
            # Todo: uniform sampling
        else:
            # it is attacker's turn
            # legal_moves = [{robot:indicator}]
            # print('We are expanding attackers actions')
            for move in legal_moves:
                next_node = copy.deepcopy(curr_game_pos)
                for robot in next_node:
                    next_node[robot][1] = move[robot]
                # print('adding to expansion analysis with: {}'.format(next_node))
                child = self.digraph.nodes[node]['state'].transition_function(next_node)
                self.digraph.add_node(self.node_counter,
                                      reward=0,
                                      n=0,
                                      uct=sys.maxsize,
                                      state=child)
                self.digraph.add_edge(node, self.node_counter, action=next_node)
                child_node_id.append(self.node_counter)
                self.node_counter += 1

        return child_node_id[0]

    def simulation(self, node):
        """
        Conducts a light playout from the specified node
        :return: The reward obtained once a terminal state is reached
        """
        # reward should be defined

        random_policy = RandomPolicy()
        # current game state
        current_state = self.digraph.nodes[node]['state']
        while not current_state.is_terminal():
            # next_node is a list/dict or empty
            next_node = random_policy.move(current_state)
            if not next_node:
                break
            # argument for transition function is next_node
            # {robot:[position, attack_indicator]}
            current_state = current_state.transition_function(next_node)
        return current_state.collected_reward()

    def backpropagation(self, last_visited, reward):
        '''
        Walk the path upwards to the root, incrementing the 'n' 
        and 'reward' attributes of the nodes along the way
        '''
        current = last_visited
        while True:
            self.digraph.nodes[current]['n'] += 1
            self.digraph.nodes[current]['reward'] += reward

            # carefully check the following statement
            # Todo
            if not list(self.digraph.predecessors(current)):
                break
            else:
                try:
                    # Todo
                    current = list(self.digraph.predecessors(current))[0]
                except IndexError:
                    break

        for node in self.digraph.nodes:
            # this node has one parent
            # print('node {} parent is {}'.format(node, self.digraph.predecessors(node)))
            if list(self.digraph.predecessors(node)):
                parent = list(self.digraph.predecessors(node))[0]
                # print('updating {} uct'.format(node))
                # print('before update uct is {}'.format(self.digraph.nodes[node]['uct']))
                self.uct(node=node, parent=parent)
                # print('after update uct is {}'.format(self.digraph.nodes[node]['uct']))

    def uct(self, node, parent):
        """
        Returns the expected value of a state, calculated as a weighted sum of
        its exploitation value and exploration value
        """
        n = self.digraph.nodes[node]['n']  # Number of plays from this node
        # total reward generated passing through this node
        # keep track of average is better
        reward = self.digraph.nodes[node]['reward']
        # number of times the parent node has been visited
        N = self.digraph.nodes[parent]['n']
        c = self.uct_c
        epsilon = self.EPSILON

        exploitation_value = reward / (n + epsilon)
        exploration_value = 2.0 * c * np.sqrt(2 * np.log(N) / (n + epsilon))
        if self.digraph.nodes[parent]['state'].turn == 'robot':
            value = exploitation_value + exploration_value
        else:
            value = -exploitation_value + exploration_value
        self.digraph.nodes[node]['uct'] = value

        return value
