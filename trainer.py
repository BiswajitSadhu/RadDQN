import logging
import numpy as np
import torch
import random
from collections import deque
import copy
from Gridworld_v6static_3sources import Gridworld
import _pickle as cPickle
from tqdm import tqdm


def addtuple(a, b):
    return tuple([sum(x) for x in zip(a, b)])


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class Trainer:

    def __init__(self, model, device, config, logdir=None):

        self._device = device
        self.config = config
        self.mode = config["mode"]
        self.logdir = logdir
        self.winratio_at_last_ten_epi = self.config["winratio_at_last_ten_epi"]
        self.model = model.to(device)
        self.model2 = copy.deepcopy(self.model)
        # self.model2.load_state_dict(self.model.state_dict())
        logging.getLogger("main").info(f"Model size: {sum(p.numel() for p in self.model.parameters())}")

        lr = float(config["learning_rate"])
        logging.getLogger("main").info(f"learning rate: {lr}")
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

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

    def train_step(self):
        # global epsilon
        self.model.train()
        mode = self.mode
        epsilon_ = self.config["epsilon"]

        total_moves = 0
        total_unsync_moves = 0
        num_epochs = self.config["num_epochs"]
        replay = deque(maxlen=self.config["mem_size"])
        print("num_epochs:", num_epochs)
        # for i in tqdm(range(num_epochs)):

        for i in range(num_epochs):
            game = Gridworld(size=10, mode=mode)
            self.S1pos = game.board.components['Source1'].pos
            self.S2pos = game.board.components['Source2'].pos
            self.S3pos = game.board.components['Source3'].pos
            state_ = game.board.render_np().reshape(1, 500)
            state1 = torch.from_numpy(state_).float()
            # game on
            status = 1
            total_reward = 0
            moves = 0
            actions = []
            actions_strings = []
            displays = []
            # visited_pos = [game.board.components['Player'].pos]
            visited_pos = []
            # cumulative_reward = 0
            old_reward = 0

            while status == 1:
                # old_c = moves
                old_pos = game.board.components['Player'].pos
                visited_pos.append(old_pos)
                valid_actions, invalid_actions = self.valid_action(game, moves, old_pos, visited_pos)
                # print(valid_actions, invalid_actions,visited_pos)
                qval = self.model(state1)
                qval_ = qval.data.numpy()
                # check to have at least one valid actions
                if valid_actions:
                    # print(epsilon_)
                    action_ = self.actions_for_validstep(game, self.ratio_list, epsilon_, moves, old_pos,
                                                         visited_pos, self.winratio_at_last_ten_epi, qval_,
                                                         old_reward)

                    # chose action based on index
                    action = self.action_set[action_]
                    # print(action)
                    actions_strings.append(action)
                    # make move with chosen action to have new position
                    game.makeMove(action)
                    # save last old position so that it can be used to make action invalid that leads to last old
                    # position
                    # new_pos = game.board.components['Player'].pos
                    actions += [game.board.components['Player'].pos]
                    displays.append(game.display())
                    # render state2 using frame of new positions
                    state2_ = game.board.render_np().reshape(1, 500) + np.random.rand(1, 500) / 10.0
                    state2 = torch.from_numpy(state2_).float()

                    if game.board.components['Player'].pos == game.board.components['Goal'].pos:
                        done = True
                        reward = game.reward_dosefunc_1byr2(S1=self.config["S1"], S2=self.config["S2"],
                                                            S3=self.config["S3"], vel=1)

                    else:
                        done = False
                        reward = game.reward_dosefunc_1byr2(S1=self.config["S1"], S2=self.config["S2"],
                                                            S3=self.config["S3"], vel=1)

                    old_reward = reward
                    moves += 1
                    total_reward += reward
                    total_unsync_moves += 1
                    total_moves += 1
                    # new_c = moves
                    # print('reward:', reward)
                    # total_reward += reward
                    # if len(valid_actions) == 1:
                    #    total_reward += 2*reward
                    #    total_moves += 2
                    # else:
                    #    total_reward += reward
                    #    total_moves += 1
                    # add experience
                    exp = (state1, action_, reward, state2, done)
                    replay.append(exp)
                    state1 = state2
                    if len(replay) > self.config["batch_size"]:
                        minibatch = random.sample(replay, self.config["batch_size"])
                        state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                        action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                        reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                        state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                        done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

                        Q1 = self.model(state1_batch)
                        with torch.no_grad():
                            Q2 = self.model2(state2_batch)

                        target = reward_batch + self.config["gamma"] * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                        # print(X, target.detach())
                        loss = torch.nn.MSELoss()(X, target.detach())

                        self.opt.zero_grad()
                        loss.backward()
                        self.losses.append(loss.item())
                        self.opt.step()
                        # print('synching model2')
                        self.model2, total_unsync_moves = self.sync_asperrules(i, self.ratio_list, total_unsync_moves)
                else:
                    done = False
                    # moves = 0

                if done:
                    self.win_lose_array.append((1, total_reward))
                    status = 0
                    # print('goal achieved: num_of_epi, moves, rewards:', i, moves, total_reward)
                    logging.getLogger("main").info("goal achieved:: episode_id: {num_of_epi}. moves taken: {moves}. "
                                                   "total_reward: {total_reward}".format(num_of_epi=i, moves=moves,
                                                                                         total_reward=total_reward))

                    # moves per winning episodes
                    self.moves_per_game.append(moves)
                    # rewards per winning episode
                    self.rewards_per_game.append(total_reward)
                    # actions per winning episode
                    self.actions_per_game.append(actions)

                if (valid_actions == [] and done is False) or moves > self.config["max_moves"]:
                    self.win_lose_array.append((0, total_reward))
                    self.moves_per_game.append(moves)
                    # rewards per winning episode
                    self.rewards_per_game.append(total_reward)
                    # actions per winning episode
                    self.actions_per_game.append(actions)

                    logging.getLogger("main").info("failed:: episode_id: {num_of_epi}. moves taken: {moves}. "
                                                   "total_reward: {total_reward}".format(num_of_epi=i, moves=moves,
                                                                                         total_reward=total_reward))
                    status = 0
                    moves = 0
                    # print("failed:", i, total_reward, total_moves)

                if (valid_actions == [] and done is False) or moves > self.config["max_moves"] or done is True:
                    win = [val for val in np.array(self.win_lose_array).T[0] if val == 1]
                    loss = [val for val in np.array(self.win_lose_array).T[0] if val == 0]
                    ratio = len(win) / (len(win) + len(loss))
                    self.ratio_list.append(ratio)

            self.savecpt(i)
            epsilon_ = self.eps(epsilon_)
        np.array(self.losses)
        self.save_output()
        print(self.sync_dict)

    def valid_action(self, game, moves, old_pos, visited_pos):
        invalid_actions = []
        addpos_action = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
        for ndx, act in enumerate(addpos_action):
            if moves == 0:
                # 0 is valid, 1 invalid, 2 lost game
                if game.validateMove('Player', act) != 0:
                    invalid_actions.append(ndx)

            elif moves > 0:
                # 0 is valid, 1 invalid, 2 lost game
                if game.validateMove('Player', act) != 0:
                    invalid_actions.append(ndx)
                # valid but already visited so considered as invalid
                elif game.validateMove('Player', act) == 0 and addtuple(old_pos, act) in visited_pos:
                    invalid_actions.append(ndx)

        invalid_actions = list(set(invalid_actions))
        all_actions = list(set(np.arange(0, 8, 1)))
        valid_actions = list(set(all_actions) ^ set(invalid_actions))

        return valid_actions, invalid_actions

    def actions_for_validstep(self, game, ratio_list, epsilon, moves, old_pos, visited_pos, winratio_at_last_ten_epi,
                              qval_, old_reward):
        self.moves = moves
        self.old_pos = old_pos
        self.visited_pos = visited_pos
        valid_actions, invalid_actions = self.valid_action(game, moves, old_pos, visited_pos)

        if valid_actions:
            # check if random number is greater than epsilon, if yes take random action
            if random.random() < epsilon:  # epsilon greedy move

                action_ = random.choice(valid_actions)
                action = self.action_set[action_]
                # dummy_move = addTuple(game.board.components['Player'].pos, action)
                dummy_move = game.fakeMove(action)
                # print(dummy_move)
                dummy_reward = game.reward_dosefunc_1byr2_dm(S1=self.config["S1"], S2=self.config["S2"],
                                                             S3=self.config["S3"], dummy_move=dummy_move, vel=1,
                                                             anticipated_pos=True)
                # print('dummy_move,dummy_reward',dummy_move, dummy_reward)

                if moves > 0 and self.config["partially_blind"] is False:
                    # print("blind expo restricted, it follows the prediction from neural network")
                    # print('dummy_move,dummy_reward,old_reward',dummy_move, dummy_reward,old_reward)
                    if dummy_reward < old_reward:
                        # print("blind expo restrited")
                        # qval predicted move: check if agent is at any of the boundary cells,
                        # if yes, choose only the valid positions
                        if invalid_actions:
                            # max(old_pos) == (game.board.size-1) or min(old_pos) == 0:
                            for each in invalid_actions:
                                qval_[0][each] = np.nan
                                action_ = np.nanargmax(qval_)
                        # all 8 actions are valid; there is no invalid actions
                        else:
                            action_ = np.argmax(qval_)  # move based on Qval
                    else:
                        # print("random")
                        action_ = action_

                elif moves > 0 and self.config["partially_blind"] is True:
                    # print("blind expo partially open if not much winning is happening,
                    # otherwise, it follows the prediction from neural network")
                    self.winratio_at_last_ten_epi = self.config["winratio_at_last_ten_epi"]
                    if len(ratio_list) > 10:
                        ratio_at_last_ten_epi = len(
                            [val for val in np.array(self.win_lose_array).T[0][-10:] if val == 1]) / 10
                        if ratio_at_last_ten_epi > winratio_at_last_ten_epi:
                            if dummy_reward < old_reward:
                                # print("blind expo restrited")
                                # qval predicted move: check if agent is at any of the boundary cells,
                                # if yes, choose only the valid positions
                                if invalid_actions:
                                    # max(old_pos) == (game.board.size-1) or min(old_pos) == 0:
                                    for each in invalid_actions:
                                        qval_[0][each] = np.nan
                                        action_ = np.nanargmax(qval_)
                                # all 8 actions are valid; there is no invalid actions
                                else:
                                    action_ = np.argmax(qval_)  # move based on Qval

                        else:
                            # random action
                            # print('taking ranodm action as win average is small in last 100 episodes')
                            action_ = action_

                    else:
                        # random action
                        action_ = action_

            else:
                # qval predicted move: check if agent is at any of the boundary cells,
                # if yes, choose only the valid positions
                if invalid_actions:
                    # max(old_pos) == (game.board.size-1) or min(old_pos) == 0:
                    for each in invalid_actions:
                        qval_[0][each] = np.nan
                        action_ = np.nanargmax(qval_)
                # all 8 actions are valid; there is no invalid actions
                else:
                    action_ = np.argmax(qval_)  # move based on Qval
        return action_

    def sync_asperrules(self, i, ratio_list, total_unsync_moves):
        # syncing model2
        config = self.config
        sync_improvised = config["sync_improvised"]
        sync_freq = config["sync_freq"]
        win = [val for val in np.array(self.win_lose_array).T[0] if val == 1]
        lost = [val for val in np.array(self.win_lose_array).T[0] if val == 0]
        ratio = len(win) / (len(win) + len(lost))
        # case I: regular sync (if sync_improvisation is false) (reg_sync)
        # case II: improvised fast sync with better win% (applies for first 5 episodes) (improv_fast_lt_5)
        # case III: improvised slow sync (applies for first 5 episodes) (improv_slow_lt_5)
        # case IV: improvised fast sync with better win% and better number of moves (applies after first 5 episodes) (improv_fast_gt_5)
        # case V: improvised slow sync (applies after first 5 episodes) (improv_slow_gt_5)
        # case VI: no sync in an episode (no_sync)
        if sync_improvised:
            if len(ratio_list) > config["min_num_epi_for_sync_rule_application"]:
                # once number of moves summing all episodes crosses sync_freq limit, the model2 gets synced with parent
                # model
                # ratio_list[-1] < 1 avoids the situation where first episode during training is won by the agent
                if ratio_list[-1] < 1 and len(win) < config["min_num_win_epi_for_sync_rule_application"]:
                    sync_freq = sync_freq - (sync_freq * ratio)
                    if total_unsync_moves > sync_freq and ratio_list[-1] > np.max(ratio_list[:-1]):
                        self.model2.load_state_dict(self.model.state_dict())
                        self.sync_dict["fast_sync_lt5"].append((i,sync_freq,total_unsync_moves))
                        logging.getLogger("main").info("fast sync: win_ratio: {ratio}, sync_freq: {sync_freq} and total_unsync_moves: {total_unsync_moves} \
                        ".format(ratio=ratio, sync_freq=sync_freq, total_unsync_moves=total_unsync_moves))

                        # print('ratio and unsync moves:', ratio, sync_freq, total_unsync_moves)
                        total_unsync_moves = 0
                    elif total_unsync_moves > config["slowness_sync_pram"] * sync_freq:
                        self.model2.load_state_dict(self.model.state_dict())
                        self.sync_dict["slow_sync_lt5"].append((i,sync_freq,total_unsync_moves))
                        logging.getLogger("main").info("slow sync: win_ratio: {ratio}, sync_freq: {sync_freq} and total_unsync_moves: {total_unsync_moves} \
                                            ".format(ratio=ratio, sync_freq=sync_freq,
                                                     total_unsync_moves=total_unsync_moves))
                        # print('double_ratio and unsync moves:', ratio, sync_freq, total_unsync_moves)
                        total_unsync_moves = 0

                if ratio_list[-1] < 1 and len(win) > config["min_num_win_epi_for_sync_rule_application"]:
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
                    if total_unsync_moves > sync_freq and ratio_list[-1] > np.max(
                            ratio_list[:-1]) and mov_av < prev_mov_av:
                        self.model2.load_state_dict(self.model.state_dict())
                        self.sync_dict["fast_sync_gt5"].append((i,sync_freq,total_unsync_moves))
                        logging.getLogger("main").info(
                            "fast sync: win_ratio: {ratio}, sync_freq: {sync_freq} and total_unsync_moves: {total_unsync_moves}, avg_moves_excluding_last_ten_episodes: {prev_mov_av},avg_moves_in_last_5_wins: {mov_av}" \
                            .format(ratio=ratio, sync_freq=sync_freq, total_unsync_moves=total_unsync_moves,
                                    prev_mov_av=prev_mov_av, mov_av=mov_av))
                        # print('ratio and unsync moves_second:', ratio, sync_freq, total_unsync_moves, prev_mov_av,
                        # mov_av)
                        total_unsync_moves = 0

                    elif total_unsync_moves > config["slowness_sync_pram"] * sync_freq:
                        self.model2.load_state_dict(self.model.state_dict())
                        self.sync_dict["slow_sync_gt5"].append((i,sync_freq,total_unsync_moves))
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
                self.sync_dict["reg_sync"].append((i,sync_freq,total_unsync_moves))

        return self.model2, total_unsync_moves

    def eps(self, epsilon_):
        # self.epsilon = epsilon
        num_epochs = self.config["num_epochs"]
        if epsilon_ > 0.1:
            epsilon_ -= (1 / num_epochs)
        else:
            epsilon_ = epsilon_
        return epsilon_

    def savecpt(self, i):
        # save checkpoints at specified intervals
        num_epochs = self.config["num_epochs"]
        system_description = self.system_name()
        if i in np.arange(100, num_epochs + 100, 100):
            # Additional information
            EPOCH = i
            PATH = "%s/%s%s.pt" % (self.logdir,i, system_description)
            LOSS = self.losses

            torch.save({
                'epoch': EPOCH,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                'loss': LOSS,
            }, PATH)

    def system_name(self):

        system_description = '_ratio_wincap_lowlr_%s_s1_%s_%s_s2_%s_%s_s3_%s_%s_epochs_%s_memsize_%s_batch_size_' \
                             '%s_sync_freq_%s_exit_reward_1_r2_1_d2_restrict_blind_expl_lr_%s_%s' % \
                             (
                                 self.mode, self.config["S1"], self.S1pos, self.config["S2"], self.S2pos,
                                 self.config["S3"], self.S3pos, self.config["num_epochs"], self.config["mem_size"],
                                 self.config["batch_size"], self.config["sync_freq"], self.config["learning_rate"],
                                 self.config["act_func"])
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

        # with open(r"someobject.pickle", "rb") as input_file:
        #    e = cPickle.load(input_file)
