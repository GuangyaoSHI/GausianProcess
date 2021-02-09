import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utilities import *
from itertools import product
import pickle

starts = {0: [(0, 0), 0], 1: [(0, 0), 0], 2: [(0, 0), 0], 3: [(0, 0), 0]}
board = setup_game()
# who take the first step
turn = 'attacker'
horizon = 4
alpha = 2
game = GameState(board, starts, turn, horizon, alpha)

fig, axs = plt.subplots(2, 2)
for axis in list(product(list(range(2)), list(range(2)))):
    i, j = axis
    plot_reward_map(game, axs[i, j])

with open("trajectories.txt", "rb") as fp:  # Unpickling
    paths = pickle.load(fp)

T = len(paths[0])

# paths = {0: [(0, 0), (1, 0), (0, 0), (1, 0), (2, 0)],
#          1: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
#          2: [(0, 0), (1, 0), (0, 0), (1, 0), (2, 0)],
#          3: [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]}

def update(t):
    label = 'time step {0}'.format(t)
    for axis in list(product(list(range(2)), list(range(2)))):
        i, j = axis
        if t > 0:
            if paths[2 * i + j][t] == paths[2 * i + j][t-1]:
                axs[i, j].plot(paths[2 * i + j][t][0], paths[2 * i + j][t][1], 'rX', markersize=12)
                axs[i, j].set_title('robot {}'.format(2 * i + j))
                axs[i, j].set_xlabel(label)
            else:
                axs[i, j].plot(paths[2 * i + j][t][0], paths[2 * i + j][t][1], 'mD')
                axs[i, j].set_title('robot {}'.format(2 * i + j))
                axs[i, j].set_xlabel(label)
    return axs

anim = FuncAnimation(fig, update, frames=T, interval=400)
anim.save('path36.gif', dpi=80, writer='imagemagick')
plt.show()
