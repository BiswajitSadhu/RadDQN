from GridBoard import *
import numpy as np


class Gridworld:

    def __init__(self, size=10, mode='static'):
        if size >= 4:
            self.board = GridBoard(size=size)
        else:
            print("Minimum board size is 4. Initialized to size 4.")
            self.board = GridBoard(size=4)

        # Add pieces, positions will be updated later
        self.board.addPiece('Player', 'P', (9, 0))
        self.board.addPiece('Goal', '+', (0, 9))
        # self.board.addPiece('Pit','-',(2,0))
        # self.board.addPiece('Wall','W',(1,0))
        self.board.addPiece('Source1', 'S1', (0, 0))
        self.board.addPiece('Source2', 'S2', (4, 4))
        self.board.addPiece('Source3', 'S3', (8, 8))

        if mode == 'static':
            self.initGridStatic()
        elif mode == 'player':
            self.initGridPlayer()
        else:
            self.initGridRand()

    # Initialize stationary grid, all items are placed deterministically
    def initGridStatic(self):
        # Setup static pieces
        self.board.components['Player'].pos = (9, 0)  # Row, Column
        self.board.components['Goal'].pos = (0, 9)
        self.board.components['Source1'].pos = (0, 0)
        self.board.components['Source2'].pos = (4, 4)
        self.board.components['Source3'].pos = (8, 8)

    # Check if board is initialized appropriately (no overlapping pieces)
    # also remove impossible-to-win boards
    def validateBoard(self):
        valid = True

        player = self.board.components['Player']
        goal = self.board.components['Goal']
        Source1 = self.board.components['Source1']
        Source2 = self.board.components['Source2']
        Source3 = self.board.components['Source3']

        [piece for name, piece in self.board.components.items()]
        all_positions = [player.pos, goal.pos, Source1.pos, Source2.pos, Source3.pos]
        if len(all_positions) > len(set(all_positions)):
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

        if (not self.validateBoard()):
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

        if (not self.validateBoard()):
            # print('Invalid grid. Rebuilding..')
            self.initGridRand()

    def validateMove(self, piece, addpos=(0, 0)):
        outcome = 0  # 0 is valid, 1 invalid, 2 lost game
        Source1 = self.board.components['Source1'].pos
        Source2 = self.board.components['Source2'].pos
        new_pos = addTuple(self.board.components[piece].pos, addpos)
        if new_pos == Source1:
            outcome = 2  # block move, player can't move to wall

        if new_pos == Source2:
            outcome = 2  # block move, player can't move to wall
        elif max(new_pos) > (self.board.size - 1):  # if outside bounds of board
            outcome = 1
        elif min(new_pos) < 0:  # if outside bounds
            outcome = 1
        # elif new_pos == pit:
        #    outcome = 2

        return outcome

    def makeMove(self, action):
        # need to determine what object (if any) is in the new grid spot the player is moving to
        # actions in {u,d,l,r,ur,ul,dr,dl}#[0,2]
        def checkMove(addpos):
            if self.validateMove('Player', addpos) in [0, 2]:
                new_pos = addTuple(self.board.components['Player'].pos, addpos)
                self.board.movePiece('Player', new_pos)

        if action == 'u':  # up
            addpos_action = (-1, 0)
            checkMove(addpos_action)
        if action == 'd':  # down
            addpos_action = (1, 0)
            checkMove(addpos_action)
        elif action == 'l':  # left
            addpos_action = (0, -1)
            checkMove(addpos_action)
        elif action == 'r':  # right
            addpos_action = (0, 1)
            checkMove(addpos_action)
        elif action == 'ur':  # up right
            addpos_action = (-1, 1)
            checkMove(addpos_action)
        elif action == 'ul':  # up left
            addpos_action = (-1, -1)
            checkMove(addpos_action)
        elif action == 'dr':  # down right
            addpos_action = (1, 1)
            checkMove(addpos_action)
        elif action == 'dl':  # down left
            addpos_action = (1, -1)
            checkMove(addpos_action)

        else:
            pass

        return (addpos_action)

    def fakeMove(self, action):
        # need to determine what object (if any) is in the new grid spot the player is moving to
        # actions in {u,d,l,r,ur,ul,dr,dl}#[0,2]
        def checkMove(addpos):
            if self.validateMove('Player', addpos) in [0, 2]:
                r_pos = addTuple(self.board.components['Player'].pos, addpos)
            return r_pos

        if action == 'u':  # up
            addpos_action = (-1, 0)
            new_pos = checkMove(addpos_action)
        if action == 'd':  # down
            addpos_action = (1, 0)
            new_pos = checkMove(addpos_action)
        elif action == 'l':  # left
            addpos_action = (0, -1)
            new_pos = checkMove(addpos_action)
        elif action == 'r':  # right
            addpos_action = (0, 1)
            new_pos = checkMove(addpos_action)
        elif action == 'ur':  # up right
            addpos_action = (-1, 1)
            new_pos = checkMove(addpos_action)
        elif action == 'ul':  # up left
            addpos_action = (-1, -1)
            new_pos = checkMove(addpos_action)
        elif action == 'dr':  # down right
            addpos_action = (1, 1)
            new_pos = checkMove(addpos_action)
        elif action == 'dl':  # down left
            addpos_action = (1, -1)
            new_pos = checkMove(addpos_action)

        else:
            # print("yes")
            pass
            # new_pos = self.board.components['Player'].pos

        return (new_pos)

    def multivariate_gaussian(self, pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2 * np.pi) ** n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

        return np.exp(-fac / 2) / N

    def NormalizeData(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def reward_dosefunc_1byr2(self, S1, S2, S3, vel=1):
        S1 = S1
        S2 = S2
        S3 = S3
        # self.board.size
        Ppos = self.board.components['Player'].pos
        S1pos = self.board.components['Source1'].pos
        S2pos = self.board.components['Source2'].pos
        S3pos = self.board.components['Source3'].pos
        Goal = self.board.components['Goal'].pos

        x = Ppos[0]
        y = Ppos[1]
        Gdist = np.sqrt((x - Goal[0]) ** 2 + (y - Goal[1]) ** 2)

        Gdist = Gdist ** 2
        shift = 0.1
        if Ppos == Goal:
            # positive reward
            # dose = (1/Gdist-0.1)
            R1sq = (x - S1pos[0]) ** 2 + (y - S1pos[1]) ** 2
            R2sq = (x - S2pos[0]) ** 2 + (y - S2pos[1]) ** 2
            R3sq = (x - S3pos[0]) ** 2 + (y - S3pos[1]) ** 2
            # dose = -np.array([10]) ; +(1/((S1+S2)*N**2)
            dose = (1 / vel) * ((S1 / R1sq) + (S2 / R2sq) + (S3 / R3sq)) - (1 / (Gdist + shift))
            # print('yes')

        elif (x == S1pos[0] and y == S1pos[1]):
            # avoid zero division error by making agent to source distance slightly larger than zero
            R1sq = ((x + shift) - S1pos[0]) ** 2 + ((y + shift) - S1pos[1]) ** 2
            R2sq = (x - S2pos[0]) ** 2 + (y - S2pos[1]) ** 2
            R3sq = (x - S3pos[0]) ** 2 + (y - S3pos[1]) ** 2
            dose = (1 / vel) * ((S1 / R1sq) + (S2 / R2sq) + (S3 / R3sq)) - (1 / Gdist)

        elif (x == S2pos[0] and y == S2pos[1]):
            R1sq = (x - S1pos[0]) ** 2 + (y - S1pos[1]) ** 2
            R2sq = ((x + shift) - S2pos[0]) ** 2 + ((y + shift) - S2pos[1]) ** 2
            R3sq = (x - S3pos[0]) ** 2 + (y - S3pos[1]) ** 2
            # dose = (1/vel)*((S1/R1sq)+(S2/R2sq))*np.sqrt(1+(np.diff([y,x])**2))
            # Gdist = np.sqrt((x - Goal[0])**2 + (y - Goal[1])**2)
            dose = (1 / vel) * ((S1 / R1sq) + (S2 / R2sq) + (S3 / R3sq)) - (1 / Gdist)

        elif (x == S3pos[0] and y == S3pos[1]):
            R1sq = (x - S1pos[0]) ** 2 + (y - S1pos[1]) ** 2
            R2sq = (x - S2pos[0]) ** 2 + (y - S2pos[1]) ** 2
            R3sq = ((x + shift) - S3pos[0]) ** 2 + ((y + shift) - S3pos[1]) ** 2
            # dose = (1/vel)*((S1/R1sq)+(S2/R2sq))*np.sqrt(1+(np.diff([y,x])**2))
            # Gdist = np.sqrt((x - Goal[0])**2 + (y - Goal[1])**2)
            dose = (1 / vel) * ((S1 / R1sq) + (S2 / R2sq) + (S3 / R3sq)) - (1 / Gdist)

        else:
            R1sq = (x - S1pos[0]) ** 2 + (y - S1pos[1]) ** 2
            R2sq = (x - S2pos[0]) ** 2 + (y - S2pos[1]) ** 2
            R3sq = (x - S3pos[0]) ** 2 + (y - S3pos[1]) ** 2
            # dose = (1/vel)*((S1/R1sq)+(S2/R2sq))*np.sqrt(1+(np.diff([y,x])**2))
            # Gdist = np.sqrt((x - Goal[0])**2 + (y - Goal[1])**2)
            dose = (1 / vel) * ((S1 / R1sq) + (S2 / R2sq) + (S3 / R3sq)) - (1 / Gdist)

        reward = -np.array(dose)
        return (reward.item())

    def reward_dosefunc_1byr2_dm(self, S1, S2, S3, dummy_move, vel=1, anticipated_pos=True):
        S1 = S1
        S2 = S2
        S3 = S3
        N = self.board.size
        if anticipated_pos:
            Ppos = dummy_move
        else:
            Ppos = self.board.components['Player'].pos
        S1pos = self.board.components['Source1'].pos
        S2pos = self.board.components['Source2'].pos
        S3pos = self.board.components['Source3'].pos
        Goal = self.board.components['Goal'].pos

        x = Ppos[0]
        y = Ppos[1]
        Gdist = np.sqrt((x - Goal[0]) ** 2 + (y - Goal[1]) ** 2)

        Gdist = Gdist ** 2
        shift = 0.1
        if Ppos == Goal:
            # positive reward
            # dose = (1/Gdist-0.1)
            R1sq = (x - S1pos[0]) ** 2 + (y - S1pos[1]) ** 2
            R2sq = (x - S2pos[0]) ** 2 + (y - S2pos[1]) ** 2
            R3sq = (x - S3pos[0]) ** 2 + (y - S3pos[1]) ** 2
            # dose = -np.array([10]); +(1/((S1+S2)*N**2))
            dose = (1 / vel) * ((S1 / R1sq) + (S2 / R2sq) + (S3 / R3sq)) - (1 / (Gdist + shift))
            # print('yes')

        elif (x == S1pos[0] and y == S1pos[1]):
            # avoid zero division error by making agent to source distance slightly larger than zero
            R1sq = ((x + shift) - S1pos[0]) ** 2 + ((y + shift) - S1pos[1]) ** 2
            R2sq = (x - S2pos[0]) ** 2 + (y - S2pos[1]) ** 2
            R3sq = (x - S3pos[0]) ** 2 + (y - S3pos[1]) ** 2
            dose = (1 / vel) * ((S1 / R1sq) + (S2 / R2sq) + (S3 / R3sq)) - (1 / Gdist)

        elif (x == S2pos[0] and y == S2pos[1]):
            R1sq = (x - S1pos[0]) ** 2 + (y - S1pos[1]) ** 2
            R2sq = ((x + shift) - S2pos[0]) ** 2 + ((y + shift) - S2pos[1]) ** 2
            R3sq = (x - S3pos[0]) ** 2 + (y - S3pos[1]) ** 2
            # dose = (1/vel)*((S1/R1sq)+(S2/R2sq))*np.sqrt(1+(np.diff([y,x])**2))
            # Gdist = np.sqrt((x - Goal[0])**2 + (y - Goal[1])**2)
            dose = (1 / vel) * ((S1 / R1sq) + (S2 / R2sq) + (S3 / R3sq)) - (1 / Gdist)

        elif (x == S3pos[0] and y == S3pos[1]):
            R1sq = (x - S1pos[0]) ** 2 + (y - S1pos[1]) ** 2
            R2sq = (x - S2pos[0]) ** 2 + (y - S2pos[1]) ** 2
            R3sq = ((x + shift) - S3pos[0]) ** 2 + ((y + shift) - S3pos[1]) ** 2
            # dose = (1/vel)*((S1/R1sq)+(S2/R2sq))*np.sqrt(1+(np.diff([y,x])**2))
            # Gdist = np.sqrt((x - Goal[0])**2 + (y - Goal[1])**2)
            dose = (1 / vel) * ((S1 / R1sq) + (S2 / R2sq) + (S3 / R3sq)) - (1 / Gdist)

        else:
            R1sq = (x - S1pos[0]) ** 2 + (y - S1pos[1]) ** 2
            R2sq = (x - S2pos[0]) ** 2 + (y - S2pos[1]) ** 2
            R3sq = (x - S3pos[0]) ** 2 + (y - S3pos[1]) ** 2
            # dose = (1/vel)*((S1/R1sq)+(S2/R2sq))*np.sqrt(1+(np.diff([y,x])**2))
            # Gdist = np.sqrt((x - Goal[0])**2 + (y - Goal[1])**2)
            dose = (1 / vel) * ((S1 / R1sq) + (S2 / R2sq) + (S3 / R3sq)) - (1 / Gdist)

        reward = -np.array(dose)

        return (reward.item())

    def reward_dosefunc_1byr(self, S1, S2, vel=1):
        S1 = S1
        S2 = S2
        N = self.board.size
        S1pos = self.board.components['Source1'].pos
        S2pos = self.board.components['Source2'].pos
        Goal = self.board.components['Goal'].pos
        Ppos = self.board.components['Player'].pos

        x = Ppos[0]
        y = Ppos[1]
        Gdist = np.sqrt((x - Goal[0]) ** 2 + (y - Goal[1]) ** 2)

        if Ppos == Goal:
            # positive reward
            # dose = (1/Gdist-0.1)
            R1 = np.sqrt((x - S1pos[0]) ** 2 + (y - S1pos[1]) ** 2)
            R2 = np.sqrt((x - S2pos[0]) ** 2 + (y - S2pos[1]) ** 2)
            # dose = -np.array([10])
            dose = (1 / vel) * ((S1 / R1) + (S2 / R2)) - (1 / (Gdist + (1 / ((S1 + S2) * N ** 2))))
            # print('yes')

        elif (x == S1pos[0] and y == S1pos[1]):
            # avoid zero division error by making agent to source distance slightly larger than zero
            R1 = np.sqrt(((x + 0.1) - S1pos[0]) ** 2 + ((y + 0.1) - S1pos[1]) ** 2)
            R2 = np.sqrt((x - S2pos[0]) ** 2 + (y - S2pos[1]) ** 2)

            dose = (1 / vel) * ((S1 / R1) + (S2 / R2)) - (1 / Gdist)

        elif (x == S2pos[0] and y == S2pos[1]):
            R1 = np.sqrt((x - S1pos[0]) ** 2 + (y - S1pos[1]) ** 2)
            R2 = np.sqrt(((x + 0.1) - S2pos[0]) ** 2 + ((y + 0.1) - S2pos[1]) ** 2)
            # dose = (1/vel)*((S1/R1sq)+(S2/R2sq))*np.sqrt(1+(np.diff([y,x])**2))
            # Gdist = np.sqrt((x - Goal[0])**2 + (y - Goal[1])**2)
            dose = (1 / vel) * ((S1 / R1) + (S2 / R2)) - (1 / Gdist)

        else:
            R1 = np.sqrt((x - S1pos[0]) ** 2 + (y - S1pos[1]) ** 2)
            R2 = np.sqrt((x - S2pos[0]) ** 2 + (y - S2pos[1]) ** 2)
            # dose = (1/vel)*((S1/R1sq)+(S2/R2sq))*np.sqrt(1+(np.diff([y,x])**2))
            # Gdist = np.sqrt((x - Goal[0])**2 + (y - Goal[1])**2)
            dose = (1 / vel) * ((S1 / R1) + (S2 / R2)) - (1 / Gdist)

        reward = -np.array(dose)

        return (reward.item())

    def display(self):
        return self.board.render()
