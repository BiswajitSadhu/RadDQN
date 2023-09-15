from GridBoard import *
import numpy as np
from itertools import islice


class Gridworld:

    def __init__(self, config, size=10, mode='static'):
        # super().__init__()
        self.config = config
        self.mode = config["mode"]
        self.size = config["grid_dimension"]
        if size >= 4:
            self.board = GridBoard(size=size)
        else:
            print("Minimum board size is 4. Initialized to size 4.")
            self.board = GridBoard(size=4)

        # Add pieces, positions will be updated later
        self.board.addPiece('Player', str(config["agent"]), eval(config["agent_pos"]))
        self.board.addPiece('Goal', str(config["exit"]), eval(config["exit_pos"]))
        # self.board.addPiece('Pit','-',(2,0))
        # self.board.addPiece('Wall','W',(1,0))
        # print(config[sources_pos])
        for ndx, (key, pos) in enumerate(config["sources_pos"]):
            self.board.addPiece('Source%s' % (ndx+1), key, pos)

        if mode == 'static':
            self.initGridStatic()
        elif mode == 'player':
            self.initGridPlayer()
        else:
            self.initGridRand()

    # Initialize stationary grid, all items are placed deterministically
    def initGridStatic(self):

        # Add pieces, positions will be updated later
        self.board.components['Player'].pos = self.config["agent_pos"]
        self.board.components['Goal'].pos = self.config["exit_pos"]
        # self.board.addPiece('Pit','-',(2,0))
        # self.board.addPiece('Wall','W',(1,0))
        for ndx, (key, pos) in enumerate(self.config["sources_pos"].items()):
            self.board.components['Source%s' % (ndx+1)].pos = pos

    # Check if board is initialized appropriately (no overlapping pieces)
    # also remove impossible-to-win boards
    def validateBoard(self):
        valid = True

        player = eval(self.board.components['Player'])
        goal = eval(self.board.components['Goal'])
        all_pos = [piece.pos for piece in self.board.components.items()]
        # print('all_pos:',all_pos)
        # all_positions = [player.pos, goal.pos, Source1.pos, Source2.pos, Source3.pos]
        if len(all_pos) > len(set(all_pos)):
            return False

        corners = [(0, 0), (0, self.board.size), (self.board.size, 0), (self.board.size, self.board.size)]
        # if player is in corner, can it move? if goal is in corner, is it blocked?
        if player.pos in corners or goal.pos in corners:
            val_move_pl = [self.validateMove('Player', addpos) for addpos in
                           [(0, 1), (1, 0), (-1, 0), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]]
            val_move_go = [self.validateMove('Goal', addpos) for addpos in
                           [(0, 1), (1, 0), (-1, 0), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]]
            if 0 not in val_move_pl or 0 not in val_move_go:
                # print(self.display())
                # print("Invalid board. Re-initializing...")
                valid = False

        return valid

    # Initialize player in random location, but keep wall, goal and pit stationary
    def initGridPlayer(self):
        # height x width x depth (number of pieces)
        self.initGridStatic()
        # place player
        self.board.components['Player'].pos = randPair(0, self.board.size)

        if not self.validateBoard():
            # print('Invalid grid. Rebuilding..')
            self.initGridPlayer()

    # Initialize grid so that goal, pit, wall, player are all randomly placed
    def initGridRand(self):
        # height x width x depth (number of pieces)
        # self.board.components['Player'].pos = randPair(0,self.board.size)
        # self.board.components['Goal'].pos = randPair(0,self.board.size)
        self.board.components['Source1'].pos = randPair(0, self.board.size)
        # self.board.components['Source1'].pos = randPair(2,7)
        self.board.components['Source2'].pos = randPair(0, self.board.size)

        if not self.validateBoard():
            # print('Invalid grid. Rebuilding..')
            self.initGridRand()

    def reward_dosefunc_1byr2(self, vel=1):
        sources = self.config['sources']
        sum_source_strength = sum(sources.values())
        # print('sum_source_strength:', sum_source_strength)
        pos_list = []
        config = self.config
        for k, v in self.config['sources'].items():
            pos_list.append(k+'pos')

        Ppos = self.board.components['Player'].pos
        if type(Ppos) == str:
            Ppos = eval(Ppos)
        Goal = self.board.components['Goal'].pos
        if type(Goal) == str:
            Goal = eval(Goal)

        sources_pos = config['sources_pos'].items()
        sources = self.config['sources'].items()
        shift = 0.1
        RSq = 0
        for (i, d), (j, p) in zip(sources, sources_pos):
            # agent steps on source
            if Ppos[0] == p[0] and Ppos[1] == p[1]:
                RSq += d / (((Ppos[0] + shift) - p[0]) ** 2 + ((Ppos[1] + shift) - p[1]) ** 2)
                Gdist = (np.sqrt((Ppos[0] - Goal[0]) ** 2 + (Ppos[1] - Goal[1]) ** 2)) ** 2
            # agent reached exit
            elif Ppos == Goal:
                RSq += d / ((Ppos[0] - p[0]) ** 2 + (Ppos[1] - p[1]) ** 2)
                Gdist = 0
            else:
                RSq += d / ((Ppos[0] - p[0]) ** 2 + (Ppos[1] - p[1]) ** 2)
                Gdist = (np.sqrt((Ppos[0] - Goal[0]) ** 2 + (Ppos[1] - Goal[1]) ** 2)) ** 2

        if Ppos == Goal:
            dose = (1 / vel) * RSq - (1 / (Gdist + shift))
        else:
            dose = (1 / vel) * RSq - (1 / Gdist)
        reward = -np.array(dose)
        return reward.item()

    def reward_dosefunc_1byr2_dm(self, dummy_move, vel=1, anticipated_pos=True):
        config = self.config
        pos_list = []
        for k, v in config['sources'].items():
            pos_list.append(k + 'pos')
        if anticipated_pos:
            Ppos = dummy_move
        else:
            Ppos = self.board.components['Player'].pos

        if type(Ppos) == str:
            Ppos = eval(Ppos)

        Goal = self.board.components['Goal'].pos
        if type(Goal) == str:
            Goal = eval(Goal)

        shift = 0.1
        RSq = 0
        for (i, d), (j, p) in zip(config['sources'].items(), config['sources_pos'].items()):
            # agent steps on source
            if Ppos[0] == p[0] and Ppos[1] == p[1]:
                RSq += d / (((Ppos[0] + shift) - p[0]) ** 2 + ((Ppos[1] + shift) - p[1]) ** 2)
                Gdist = (np.sqrt((Ppos[0] - Goal[0]) ** 2 + (Ppos[1] - Goal[1]) ** 2)) ** 2
            # agent reached exit
            elif Ppos == Goal:
                RSq += d / ((Ppos[0] - p[0]) ** 2 + (Ppos[1] - p[1]) ** 2)
                Gdist = 0
            else:
                RSq += d / ((Ppos[0] - p[0]) ** 2 + (Ppos[1] - p[1]) ** 2)
                Gdist = (np.sqrt((Ppos[0] - Goal[0]) ** 2 + (Ppos[1] - Goal[1]) ** 2)) ** 2

        if Ppos == Goal:
            Gdist = 0
            dose = (1 / vel) * RSq - (1 / (Gdist + shift))
        else:
            dose = (1 / vel) * RSq - (1 / Gdist)
        reward = -np.array(dose)
        return reward.item()

    def display(self):
        return self.board.render()
