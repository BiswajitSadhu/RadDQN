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
from gridworld_config import Gridworld

sys.path.append(".")

# random_seeds = np.array([3007,2111,2010,0312,1802])
def seed_torch(seed=int(3007)):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="The path to the config file in yaml format. ")
    parser.add_argument("--model_type", choices=["RadDqn"], default="RadDqn")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--logdir", type=str, help="Tensorboard and model checkpoints will be saved here. ")
    parser.add_argument("--seed", type=int, help="random seed for weight initialization")
    directory = parser.parse_args().logdir
    if not os.path.exists(directory):
        os.makedirs(directory)
    return parser.parse_args()

def parse_config(configfile: str):
    with open(configfile, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main():
    args = parse_arguments()
    seed_torch(seed=int(args.seed))
    logging.basicConfig(filename='%s/info.log'% args.logdir, level=logging.INFO, filemode="w+")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger("main").info(f"cuda available: {torch.cuda.is_available()}")
    config = parse_config(args.config_file)
    if args.model_type == "RadDqn":
        model = RadDqn(config=config, device=args.device)
        RadDqn.weights_initialization(model)
        # model.apply(RadDqn.weights_initialization)
        print('model details:', model)
        logging.getLogger("main").info("Model summary: {model}".format(model=model))
        logging.getLogger("main").info("random seed: {seed}".format(seed=int(args.seed)))
        for u in model.parameters():
            print(u)
            logging.getLogger("main").info("model parameters: {parameter}".format(parameter=u))
        gridworld = Gridworld(config=config, size=10, mode='static')
        trainer = Trainer(model, gridworld, config=config, device=args.device, logdir=args.logdir)
        # print('model:', model)

        trainer.train_step()


if __name__ == "__main__":
    main()
