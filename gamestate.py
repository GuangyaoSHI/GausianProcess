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


class GameState:
    def __init__(self, G, starts, turn, horizon, alpha):
        # starts = {robot:[position, attack_indicator]}
        # define world map
        self.G = G
        self.horizon = horizon
        # turn = 'robot' or 'attacker'
        # self.turn will decide the legal_moves
        # and horizon
        # who take the first step
        self.firstTurn = turn
        # who take the second step
        self.secTurn = 'robot' if turn != 'robot' else 'attacker'
        # current turn
        self.turn = turn
        # number of total attacks
        self.ALPHA = alpha
        # number of attacks left
        self.alpha = alpha
        #        
        # trajectories/paths of robots and attackers
        # {robot:[t0,t1,...]}
        self.paths_robots = {}
        self.paths_attackers = {}
        for robot in starts:
            self.paths_robots[robot] = [starts[robot][0]]
            self.paths_attackers[robot] = [starts[robot][1]]
        # starts is a dictionary of starting positions of robots
        # starts = {robot:position}
        self.currNode = starts

    def move(self, next_node):
        # next_node is a dictionary
        # next_node = {robot:[position, attack_indicator]}
        # check whether the number of attacked robots is greater than
        # the ALPHA
        num_att = 0
        for robot in next_node:
            num_att += next_node[robot][1]
        assert num_att <= self.ALPHA, 'more than ALPHA robots are attacked'
        # update number of attacks left
        self.alpha = self.ALPHA - num_att
        for robot in self.paths_robots:
            self.paths_robots[robot].append(next_node[robot][0])
            self.paths_attackers[robot].append(next_node[robot][1])
       # change the game turn
        if self.turn != self.firstTurn:
            self.horizon -= 1
            self.turn = self.firstTurn
        else:
            self.turn = self.secTurn
        self.currNode = next_node

    def legal_moves(self, currNode):
        """
        currNode = {robot:[position, attack_indicator]}
        return the neighbors of the current node
        ***********************
        if it is robots's turn return possible actions
        {robot:[action]}
        ***********************
        if it is attacker's turn return possible attack
        [{robot:indicator},]
        ***********************
        if the robot has run out of budget, return an empty dict
        """
        if self.turn == 'robot':
            possible_moves = {}
            # check if budget has been used up
            if self.horizon <= 0:
                # print('Time up!')
                return possible_moves
            for robot in currNode:
                curr_pos = currNode[robot][0]
                # if the robot is already attacked
                # stay in the same place
                if currNode[robot][1]:
                    possible_moves[robot] = [curr_pos]
                else:
                    possible_moves[robot] = list(self.G.neighbors(curr_pos))
            return possible_moves
        else:
            possible_moves = []
            # check if budget has been used up
            if self.horizon <= 0:
                # print('Time up!')
                return possible_moves
            # attacked robots so far
            attacked = []
            # survided robots so far
            survided = []
            # attack_indicator
            move_ind = {}
            for robot in currNode:
                move_ind[robot] = currNode[robot][1]
                if currNode[robot][1] == 1:
                    attacked.append(robot)
                else:
                    survided.append(robot)
            # default choice for the attacker is to do nothing
            # the loop will add more options/actions for the attacker to choose
            # they all start from the current attack state and set some new 1's to indicate new attacks
            # even i = 0, the j loop still execute once
            # possible_moves.append(move_ind)
            for i in range(0, self.alpha + 1):
                for j in list(combinations(survided, i)):
                    # if i=0 j=() empty tuple
                    # current attack indicator
                    move = copy.deepcopy(move_ind)
                    # each j is a tuple of robots to be attacked
                    # for example if i = 2, j may be like (2,3)
                    for r in j:
                        # each robot in tuple j will be attacked
                        move[r] = 1
                    possible_moves.append(move)
            return possible_moves

    def transition_function(self, next_node):
        # verify that the next position is legal
        # next_node = {robot:[position, attack_indicator]}
        if self.turn == 'robot':
            for robot in next_node:
                assert next_node[robot][0] in self.legal_moves(self.currNode)[robot]
        else:
            attack_move = {}
            for robot in next_node:
                attack_move[robot] = next_node[robot][1]
            assert attack_move in self.legal_moves(self.currNode)
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
        total = 0
        nodes = []
        for robot in self.paths_robots:
            nodes += (self.paths_robots[robot])
        nodes = list(set(nodes))
        for node in nodes:
            total += self.G.nodes[node]['reward']
        return total


def test(starts=[(0, 0), (4, 0)]):
    gamestate = GameState(starts)
    print(gamestate.__str__())
    legal_moves = gamestate.legal_moves(starts)
    gamestate.move((1, 0))
    print(gamestate.__str__())
    pos = {}
    labels = {}
    for node in gamestate.G.nodes:
        pos[node] = node
        labels[node] = gamestate.G.nodes[node]['reward']
    nx.draw(gamestate.G, pos=pos, with_labels=True)
