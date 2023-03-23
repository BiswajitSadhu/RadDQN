import torch
# import Gridworld_v6static_3sources
# from Gridworld_v6static_3sources import Gridworld
import numpy as np
import random
import os


class RadDqn(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.mode = config["mode"]
        # self.act_func = config["leaky"]
        self.S1 = config["S1"]
        self.S2 = config["S2"]
        self.S3 = config["S3"]
        self.mem_size = config["mem_size"]
        self.num_epochs = config["num_epochs"]
        self.max_moves = config["max_moves"]
        self.batch_size = config["batch_size"]
        self.sync_freq = config["sync_freq"]

        # game = Gridworld(size=10, mode=mode)

        # construct the atom decoder:
        layers = []
        layerwisesize = config["layerwisesize"]
        in_features = 500
        for out_size in layerwisesize:
            layers.append(torch.nn.Linear(in_features=in_features, out_features=out_size))
            layers.append(torch.nn.LeakyReLU())
            in_features = out_size
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

    def weights_initialization(self):
        """
        When we define all the modules such as the layers in '__init__()'
        method above, these are all stored in 'self.modules()'.
        We go through each module one by one. This is the entire network,
        basically.
        """

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

