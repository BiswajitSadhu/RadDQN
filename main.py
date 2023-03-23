import sys
import os
import random
import numpy as np
from models.raddqn import RadDqn
from trainer import Trainer
import argparse
import torch
import logging
import yaml

sys.path.append(".")


def seed_torch(seed=int(3007)):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", required=True, help="The path to the config file in yaml format. ")
    parser.add_argument("--model_type", choices=["RadDqn"], default="RadDqn")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--logdir", type=str, help="Tensorboard and model checkpoints will be saved here. ")
    directory = parser.parse_args().logdir
    if not os.path.exists(directory):
        os.makedirs(directory)
    return parser.parse_args()

def parse_config(configfile: str):
    with open(configfile, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main():
    args = parse_arguments()
    logging.basicConfig(filename='%s/info.log'% args.logdir, level=logging.INFO, filemode="w+")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger("main").info(f"cuda available: {torch.cuda.is_available()}")
    config = parse_config(args.config_file)
    if args.model_type == "RadDqn":

        model = RadDqn(config=config, device=args.device)
        # for u in model.parameters():
        #    print('SS', u)
        RadDqn.weights_initialization(model)
        print(model)
        # model.apply(RadDqn.weights_initialization)
        for u in model.parameters():
            print(u)

        trainer = Trainer(model, config=config, device=args.device, logdir=args.logdir)
        # print('model:', model)

        trainer.train_step()


if __name__ == "__main__":
    main()
