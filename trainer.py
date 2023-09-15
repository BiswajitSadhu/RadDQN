import logging
import numpy as np
import torch
import random
from collections import deque
import copy
from gridworld_config import Gridworld
import _pickle as cPickle
from tqdm import tqdm


# def addtuple(a, b):
#    return tuple([sum(x) for x in zip(a, b)])

def addtuple(a, b):
    if type(a) == str:
        a = eval(a)
    if type(b) == str:
        b = eval(b)
    return tuple([sum(x) for x in zip(a, b)])


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class Trainer:

    def __init__(self, model, gridworld, device, config, logdir=None):
        super().__init__()
        self.gridworld = gridworld
        self._device = device
        self.config = config
        self.mode = config["mode"]
        self.logdir = logdir
        self.winratio_at_last_ten_epi = self.config["winratio_at_last_ten_epi"]
        self.model = model.to(device)
        self.model2 = copy.deepcopy(self.model)
        logging.getLogger("main").info(f"Model size: {sum(p.numel() for p in self.model.parameters())}")
        lr = float(config["learning_rate"])
        logging.getLogger("main").info(f"learning rate: {lr}")
        logging.getLogger("main").info(f"config: {config}")
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.in_features = config["num_pieces"] * (config['grid_dimension'] ** 2)
        self.action_set = {
            0: 'u',
            1: 'd',
            2: 'l',
            3: 'r',
            4: 'ur',
            5: 'ul',
            6: 'dr',
            7: 'dl',
        }
        self.addpos_action = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
        self.losses = []
        self.rewards_per_game = []
        self.moves_per_game = []
        self.actions_per_game = []
        self.win_lose_array = []
        self.ratio_list = []
        self.sync_dict = {}
        self.sync_dict.setdefault('reg_sync', [])
        self.sync_dict.setdefault('fast_sync_lt5', [])
        self.sync_dict.setdefault('slow_sync_lt5', [])
        self.sync_dict.setdefault('fast_sync_gt5', [])
        self.sync_dict.setdefault('slow_sync_gt5', [])
        self.random_actions = []

        logging.getLogger("main").info("system description:: \n{sys_desc}".format(sys_desc=self.system_name()))

    def train_step(self):
        self.model.train()
        mode = self.mode
        epsilon_ = self.config["epsilon"]
        total_moves = 0
        total_unsync_moves = 0
        num_epochs = self.config["num_epochs"]
        replay = deque(maxlen=self.config["mem_size"])
        print("num_epochs:", num_epochs)
        exit_pos = eval(self.config["exit_pos"])

        # for i in tqdm(range(num_epochs)):
        for i in range(num_epochs):
            game = Gridworld(size=self.config["grid_dimension"], mode=mode, config=self.config)
            if i == 0:
                logging.getLogger("main").info("2D gridworld in display: \n{display}".format(display=game.display()))
            state_ = game.board.render_np().reshape(1, int(self.in_features))
            state1 = torch.from_numpy(state_).float()
            # game on
            status = 1
            done = False
            total_reward = 0
            moves = 0
            random_step_list = []
            actions_strings = []
            visited_pos = []
            old_reward = 0
            reward_list = []
            while status == 1:
                current_pos = game.board.components['Player'].pos
                if type(current_pos) == str:
                    current_pos = eval(game.board.components['Player'].pos)
                else:
                    current_pos = game.board.components['Player'].pos
                visited_pos.append(current_pos)
                qval = self.model(state1)
                qval_ = qval.data.numpy()
                valid_actions_, invalid_actions_ = self.valid_action(moves, current_pos, visited_pos)
                # print(valid_actions_, invalid_actions_, visited_pos, current_pos, qval_)
                if valid_actions_ and not done:
                    action_, random_step_or_not, random_step_type = self.actions_for_validstep(game, self.ratio_list, epsilon_,
                                                                             moves, current_pos, qval_, old_reward,
                                                                             valid_actions_, invalid_actions_)
                    # chose action based on index
                    action = self.action_set[action_]
                    actions_strings.append(action)
                    random_step_list.append((current_pos,random_step_or_not,random_step_type))
                    # make move with chosen action to have new position
                    self.make_move(game, current_pos, action_)
                    # render state2 using frame of new positions
                    state2_ = game.board.render_np().reshape(1, self.in_features) + np.random.rand(1,
                                                                                                   self.in_features) / \
                              self.config["grid_dimension"]
                    state2 = torch.from_numpy(state2_).float()

                    if game.board.components['Player'].pos == exit_pos:

                        done = True
                        reward = game.reward_dosefunc_1byr2(vel=1)
                        status = 0
                    else:
                        done = False
                        reward = game.reward_dosefunc_1byr2(vel=1)

                    old_reward = reward
                    moves += 1
                    total_reward += reward
                    reward_list.append(reward)
                    total_unsync_moves += 1
                    total_moves += 1
                    # add experience
                    exp = (state1, action_, reward, state2, done)
                    replay.append(exp)
                    state1 = state2
                    if len(replay) > self.config["batch_size"]:
                        mini_batch = random.sample(replay, self.config["batch_size"])
                        state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in mini_batch])
                        action_batch = torch.Tensor([a for (s1, a, r, s2, d) in mini_batch])
                        reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in mini_batch])
                        state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in mini_batch])
                        done_batch = torch.Tensor([d for (s1, a, r, s2, d) in mini_batch])
                        Q1 = self.model(state1_batch)
                        with torch.no_grad():
                            Q2 = self.model2(state2_batch)
                        target = reward_batch + self.config["gamma"] * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                        loss = torch.nn.MSELoss()(X, target.detach())
                        self.opt.zero_grad()
                        loss.backward()
                        self.losses.append(loss.item())
                        self.opt.step()
                        self.model2, total_unsync_moves = self.sync_asperrules(i, total_unsync_moves)

                if done:
                    status = 0
                    logging.getLogger("main").info("goal achieved:: episode_id: {num_of_epi}. moves taken: {moves}. "
                                                   "total_reward: {total_reward}".format(num_of_epi=i, moves=moves,
                                                                                         total_reward=total_reward))

                    logging.getLogger("reward_per_epoch").info(
                        "reward per epoch:: \n{r}".format(r=reward_list))
                    ratio, win = self.ratio()

                    # win = 1, lost = 0
                    self.win_lose_array.append((1, total_reward))
                    # win/total episode
                    self.ratio_list.append(ratio)
                    # random action = 1 else 0
                    self.random_actions.append(random_step_list)
                    # moves per winning episodes
                    self.moves_per_game.append(moves)
                    # rewards per winning episode
                    self.rewards_per_game.append(total_reward)
                    # actions (pos) per winning episode
                    self.actions_per_game.append(visited_pos)

                if (valid_actions_ == [] and done is False) or (moves > self.config["max_moves"]):
                    self.win_lose_array.append((0, total_reward))
                    self.moves_per_game.append(moves)
                    # rewards per winning episode
                    self.rewards_per_game.append(total_reward)
                    # actions per winning episode
                    self.actions_per_game.append(visited_pos)
                    ratio, win = self.ratio()
                    self.ratio_list.append(ratio)
                    # random action = 1 else 0
                    self.random_actions.append(random_step_list)
                    logging.getLogger("main").info("failed:: episode_id: {num_of_epi}. moves taken: {moves}. "
                                                   "total_reward: {total_reward}".format(num_of_epi=i, moves=moves,
                                                                                         total_reward=total_reward))
                    status = 0
                    moves = 0

            self.savecpt(i)
            epsilon_ = self.eps(epsilon_)
        np.array(self.losses)
        self.save_output()
        logging.getLogger("main").info("sync_dict: {sync_dict}".format(sync_dict=self.sync_dict))
        logging.getLogger("main").info("win ratio_list: {ratio_list}".format(ratio_list=self.ratio_list))

    def validatemove(self, current_pos, addpos=(0, 0)):
        outcome = 0  # 0 is valid, 1 invalid, 2 lost game
        new_pos = addtuple(current_pos, addpos)

        if max(new_pos) > (self.config["grid_dimension"] - 1):  # if outside bounds of board
            outcome = 1

        elif min(new_pos) < 0:  # if outside bounds
            outcome = 1

        elif new_pos in tuple([tuple(val) for val in self.config["sources_pos"].values()]):
            outcome = 2

        return outcome

    def fake_move(self, current_pos, action_):
        act = self.addpos_action[action_]
        # take a fake move to get sneak peek at future position
        # outcome should be 0,2; 1 is not considered as it indicates invalid position
        # outcome 2 is when agent steps on source

        if type(current_pos) == str:
            current_pos = eval(current_pos)

        if self.validatemove(current_pos, act) in [0, 2]:
            dummy_pos_ = addtuple(current_pos, act)

        return dummy_pos_

    def make_move(self, game, current_pos, action_):
        act = self.addpos_action[action_]
        # take a move and update position
        # outcome should be 0,2; 1 is not considered as it indicates invalid position
        # outcome 2 is when agent steps on source

        if type(current_pos) == str:
            current_pos = eval(current_pos)

        if self.validatemove(current_pos, act) in [0, 2]:
            updated_pos_ = addtuple(current_pos, act)
            game.board.movePiece('Player', updated_pos_)

        return

    def valid_action(self, moves, current_pos, visited_pos):
        invalid_actions = []
        for ndx, act in enumerate(self.addpos_action):
            if moves == 0:
                # 0 is valid, 1 invalid, 2 lost game
                if self.validatemove(current_pos, act) != 0:
                    invalid_actions.append(ndx)

            elif moves > 0:
                # 0 is valid, 1 invalid, 2 lost game
                # valid but already visited so considered as invalid
                if self.validatemove(current_pos, act) == 0:
                    dummy_pos = addtuple(current_pos, act)
                    if dummy_pos in visited_pos:
                        # print('oo:', current_pos, act, dummy_pos, visited_pos)
                        invalid_actions.append(ndx)
                elif self.validatemove(current_pos, act) != 0:
                    invalid_actions.append(ndx)

        invalid_actions = list(set(invalid_actions))
        all_actions = list(set(np.arange(0, 8, 1)))
        valid_actions = list(set(all_actions) ^ set(invalid_actions))

        return valid_actions, invalid_actions

    def actions_for_validstep(self, game, ratio_list, epsilon, moves, current_pos,
                              qval_, old_reward, valid_actions, invalid_actions):

        random_step = 1  # 1 is random; 0 is not random
        random_step_type = 'v'
        # nr: nor random; p: partly restricted, f: fully restricted;
        # pv and fv: random despite restriction on blind exploration.
        # check if random number is greater than epsilon, if yes take random action
        if random.random() < epsilon:  # epsilon greedy move
            action_ = random.choice(valid_actions)
            # action = self.action_set[action_]
            # dummy_move = addTuple(game.board.components['Player'].pos, action)
            dummy_pos = self.fake_move(current_pos, action_)
            # dummy_move = game.fakeMove(action)
            # print(dummy_move)
            dummy_reward = game.reward_dosefunc_1byr2_dm(dummy_move=dummy_pos, vel=1, anticipated_pos=True)
            if self.config["vanilla_exploration"] is True:
                action_ = action_
                random_step = 1
                random_step_type = 'v'

            # action is determined by neural net: (a) partially blind (b) fully blind
            elif not self.config["vanilla_exploration"]:
                if self.config["partially_blind"]:
                    if moves > 0:
                        # print("blind expo partially open if not much winning is happening,
                        # otherwise, it follows the prediction from neural network")
                        if len(ratio_list) > 10:
                            ratio_at_last_ten_epi = len(
                                [val for val in np.array(self.win_lose_array).T[0][-10:] if val == 1]) / 10
                            if ratio_at_last_ten_epi > self.config["winratio_at_last_ten_epi"]:
                                # check next reward; if it is less than reward at present state ask neural net
                                # for choosing action; else take random action
                                if dummy_reward < old_reward:
                                    # print("blind expo restricted")
                                    # qval predicted move: check if agent is at any of the boundary cells,
                                    # if yes, choose only the valid positions
                                    if invalid_actions:
                                        for each in invalid_actions:
                                            qval_[0][each] = np.nan
                                            action_ = np.nanargmax(qval_)
                                            random_step = 0
                                            random_step_type = 'p'
                                    # all 8 actions are valid; there is no invalid actions
                                    else:
                                        action_ = np.argmax(qval_)  # move based on Qval
                                        random_step = 0
                                        random_step_type = 'p'
                                else:
                                    action_ = action_
                                    random_step = 1
                                    random_step_type = 'pv'
                            else:
                                action_ = action_
                                random_step = 1
                                random_step_type = 'pv'
                        else:
                            action_ = action_
                            random_step = 1
                            random_step_type = 'pv'

                if not self.config["partially_blind"]:
                    if moves > 0:
                        # with no condition on winning ratio
                        # check next reward; if it is less than reward at present state ask neural net
                        # for choosing action; else take random action
                        if dummy_reward < old_reward:
                            # print("blind expo restricted")
                            # qval predicted move: check if agent is at any of the boundary cells,
                            # if yes, choose only the valid positions
                            if invalid_actions:
                                # max(current_pos) == (game.board.size-1) or min(current_pos) == 0:
                                for each in invalid_actions:
                                    qval_[0][each] = np.nan
                                    action_ = np.nanargmax(qval_)
                                    random_step = 0
                                    random_step_type = 'f'
                            # all 8 actions are valid; there is no invalid actions
                            else:
                                action_ = np.argmax(qval_)  # move based on Qval
                                random_step = 0
                                random_step_type = 'f'
                        else:
                            # random action
                            action_ = action_
                            random_step = 1
                            random_step_type = 'fv'
                    else:
                        # random action
                        action_ = action_
                        random_step = 1
                        random_step_type = 'fv'

        else:
            # ask neural net for action as random.random() > epsilon
            if invalid_actions:
                # max(current_pos) == (game.board.size-1) or min(current_pos) == 0:
                for each in invalid_actions:
                    qval_[0][each] = np.nan
                    action_ = np.nanargmax(qval_)
                    random_step = 0
                    random_step_type = 'nr'
            # all 8 actions are valid; there is no invalid actions
            else:
                action_ = np.argmax(qval_)  # move based on Qval
                random_step = 0
                random_step_type = 'nr'

        return action_, random_step, random_step_type

    def sync_asperrules(self, i, total_unsync_moves):
        # syncing model2
        config = self.config
        sync_improvised = config["sync_improvised"]
        sync_freq = config["sync_freq"]
        ratio, win = self.ratio()
        # case I: regular sync (if sync_improvisation is false) (reg_sync)
        # case II: improvised fast sync with better win% (applies for first 5 episodes) (improv_fast_lt_5)
        # case III: improvised slow sync (applies for first 5 episodes) (improv_slow_lt_5)
        # case IV: improvised fast sync with better win% and better number of moves (applies after first 5 episodes) (improv_fast_gt_5)
        # case V: improvised slow sync (applies after first 5 episodes) (improv_slow_gt_5)
        # case VI: no sync in an episode (no_sync)
        if sync_improvised:
            # print(self.ratio_list, sync_freq, config["min_num_epi_for_sync_rule_application"])
            if len(self.ratio_list) > config["min_num_epi_for_sync_rule_application"]:
                # print(ratio_list,sync_freq)
                # once number of moves summing all episodes crosses sync_freq limit, the model2 gets synced with parent
                # model
                # ratio_list[-1] < 1 avoids the situation where first episode during training is won by the agent
                if self.ratio_list[-1] < 1 and len(win) < config["min_num_win_epi_for_sync_rule_application"]:
                    sync_freq = sync_freq - (sync_freq * ratio)
                    if total_unsync_moves > sync_freq and self.ratio_list[-1] > np.max(self.ratio_list[:-1]):
                        self.model2.load_state_dict(self.model.state_dict())
                        self.sync_dict["fast_sync_lt5"].append((i, sync_freq, total_unsync_moves))
                        logging.getLogger("main").info("fast sync: win_ratio: {ratio}, sync_freq: {sync_freq} and total_unsync_moves: {total_unsync_moves} \
                        ".format(ratio=ratio, sync_freq=sync_freq, total_unsync_moves=total_unsync_moves))

                        # print('ratio and unsync moves:', ratio, sync_freq, total_unsync_moves)
                        total_unsync_moves = 0

                    if total_unsync_moves > config["slowness_sync_pram"] * sync_freq:
                        self.model2.load_state_dict(self.model.state_dict())
                        self.sync_dict["slow_sync_lt5"].append((i, sync_freq, total_unsync_moves))
                        logging.getLogger("main").info("slow sync: win_ratio: {ratio}, sync_freq: {sync_freq} and total_unsync_moves: {total_unsync_moves} \
                                            ".format(ratio=ratio, sync_freq=sync_freq,
                                                     total_unsync_moves=total_unsync_moves))
                        # print('double_ratio and unsync moves:', ratio, sync_freq, total_unsync_moves)
                        total_unsync_moves = 0

                if self.ratio_list[-1] < 1 and len(win) > config["min_num_win_epi_for_sync_rule_application"]:
                    min_num_win_epi = config["min_num_win_epi_for_sync_rule_application"]
                    excluding_last_n_epi_for_average_prev_moves = config["excluding_last_n_epi_for_average_prev_moves"]
                    sync_freq = sync_freq - (sync_freq * ratio)
                    win_moves = [self.moves_per_game[vdx] for vdx, val in enumerate(np.array(self.win_lose_array).T[0])
                                 if
                                 val == 1]
                    prev_mov_av = moving_average(self.moves_per_game[:-excluding_last_n_epi_for_average_prev_moves],
                                                 n=(
                                                         len(self.moves_per_game) - excluding_last_n_epi_for_average_prev_moves))[
                        0]
                    # mov_av = moving_average(moves_per_game[-10:],n=10)[0]
                    # prev_mov_av = moving_average(win_moves[:-5],n=len(win_moves)-5)[0]
                    mov_av = moving_average(win_moves[-min_num_win_epi:], n=min_num_win_epi)[0]
                    if total_unsync_moves > sync_freq and self.ratio_list[-1] > np.max(
                            self.ratio_list[:-1]) and mov_av < prev_mov_av:
                        self.model2.load_state_dict(self.model.state_dict())
                        self.sync_dict["fast_sync_gt5"].append((i, sync_freq, total_unsync_moves))
                        logging.getLogger("main").info(
                            "fast sync: win_ratio: {ratio}, sync_freq: {sync_freq} and total_unsync_moves: {total_unsync_moves}, avg_moves_excluding_last_ten_episodes: {prev_mov_av},avg_moves_in_last_5_wins: {mov_av}" \
                                .format(ratio=ratio, sync_freq=sync_freq, total_unsync_moves=total_unsync_moves,
                                        prev_mov_av=prev_mov_av, mov_av=mov_av))
                        # print('ratio and unsync moves_second:', ratio, sync_freq, total_unsync_moves, prev_mov_av,
                        # mov_av)
                        total_unsync_moves = 0

                    elif total_unsync_moves > config["slowness_sync_pram"] * sync_freq:

                        self.model2.load_state_dict(self.model.state_dict())
                        self.sync_dict["slow_sync_gt5"].append((i, sync_freq, total_unsync_moves))
                        logging.getLogger("main").info("slow sync: win_ratio: {ratio}, sync_freq: {sync_freq} and total_unsync_moves: {total_unsync_moves} \
                                                                ".format(ratio=ratio, sync_freq=sync_freq,
                                                                         total_unsync_moves=total_unsync_moves))
                        # print('double_ratio and unsync moves_second:', ratio, sync_freq, total_unsync_moves)
                        total_unsync_moves = 0

        else:
            if total_unsync_moves % sync_freq == 0:
                logging.getLogger("main").info("syncing done without improvisation: win_ratio: {ratio}, sync_freq: {sync_freq} and total_unsync_moves: {total_unsync_moves} \
                                                                                ".format(ratio=ratio,
                                                                                         sync_freq=sync_freq,
                                                                                         total_unsync_moves=total_unsync_moves))
                # print("syncing done without improvisation")
                self.model2.load_state_dict(self.model.state_dict())
                self.sync_dict["reg_sync"].append((i, sync_freq, total_unsync_moves))
                total_unsync_moves = 0

        return self.model2, total_unsync_moves

    def eps(self, epsilon_):
        # self.epsilon = epsilon
        num_epochs = self.config["num_epochs"]
        if epsilon_ > 0.1:
            epsilon_ -= (1 / num_epochs)
        else:
            epsilon_ = epsilon_
        return epsilon_

    def ratio(self):
        if self.win_lose_array:
            win = [val for val in np.array(self.win_lose_array).T[0] if val == 1]
            lost = [val for val in np.array(self.win_lose_array).T[0] if val == 0]
            if not win:
                len_win = 0
                ratio = len_win / (len_win + len(lost))
                # print('rat:',ratio)
            elif not lost and len(win) > 0:
                len_lost = 0
                ratio = len(win) / (len(win) + len_lost)
            elif not win and not lost:
                ratio = 0
            else:
                ratio = len(win) / (len(win) + len(lost))
        else:
            ratio = 0
            win = [0]

        return (ratio, win)

    def savecpt(self, i):
        # save checkpoints at specified intervals
        num_epochs = self.config["num_epochs"]
        system_description = self.system_name()
        if i in np.arange(100, num_epochs + 100, 100):
            # Additional information
            EPOCH = i
            PATH = "%s/%s%s.pt" % (self.logdir, i, system_description)
            LOSS = self.losses

            torch.save({
                'epoch': EPOCH,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                'loss': LOSS,
            }, PATH)

    def system_name(self):
        sname = ''
        for (i, d), (j, s) in zip(self.config["sources"].items(), self.config["sources_pos"].items()):
            sname += '_%s_%s_%s_%s' % (i, d, s[0], s[1])

        system_description = '%s_%s_epochs_%s_memsize_%s_batch_size_' \
                             '%s_sync_freq_%s_lr_%s_func_%s_sync_improv_%s_vanilla_exploration_%s_partially_blind_%s' % \
                             (
                                 self.mode, self.config["num_epochs"], self.config["mem_size"],
                                 self.config["batch_size"], self.config["sync_freq"], self.config["learning_rate"],
                                 self.config["act_func"], sname, self.config["sync_improvised"],
                                 self.config["vanilla_exploration"], self.config["partially_blind"])
        return system_description

    def save_output(self):

        system_description = self.system_name()

        with open(r"%s/losses_%s_.pickle" % (self.logdir, system_description), "wb") as ofa:
            cPickle.dump(self.losses, ofa)

        with open(r"%s/rewards_per_game_%s_.pickle" % (self.logdir, system_description), "wb") as ofb:
            cPickle.dump(self.rewards_per_game, ofb)

        with open(r"%s/moves_per_game_%s_.pickle" % (self.logdir, system_description), "wb") as ofc:
            cPickle.dump(self.moves_per_game, ofc)

        with open(r"%s/actions_per_game_%s_.pickle" % (self.logdir, system_description), "wb") as ofd:
            cPickle.dump(self.actions_per_game, ofd)

        with open(r"%s/win_lose_array_%s_.pickle" % (self.logdir, system_description), "wb") as ofe:
            cPickle.dump(self.win_lose_array, ofe)

        with open(r"%s/ratio_list_%s_.pickle" % (self.logdir, system_description), "wb") as off:
            cPickle.dump(self.ratio_list, off)

        with open(r"%s/sync_info_%s_.pickle" % (self.logdir, system_description), "wb") as ofg:
            cPickle.dump(self.sync_dict, ofg)

        with open(r"%s/random_actions_info_%s_.pickle" % (self.logdir, system_description), "wb") as ofh:
            cPickle.dump(self.random_actions, ofh)

        # with open(r"someobject.pickle", "rb") as input_file:
        #    e = cPickle.load(input_file)
