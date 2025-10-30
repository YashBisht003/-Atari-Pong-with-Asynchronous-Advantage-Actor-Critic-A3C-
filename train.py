import torch
import torch.optim as optim
import torch.multiprocessing as mp
import gym
import os
import logging
import time
import visdom
import cv2
from environment import AtariEnv
from A3C import *
import my_optim
from multiprocessing import Value
import ale_py
import argparse

class A3CAtari(object):

    def __init__(self, args_, logger_):
        self.args = args_
        self.logger = logger_
        self.env = AtariEnv(gym.make(self.args.game), args_.frame_seq, args_.frame_skip, render=True)
        self.shared_model = A3CLSTMNet(self.env.state_shape, self.env.action_dim)
        self.shared_model.share_memory()  # Ensures shared model is in shared memory
        self.optim = my_optim.SharedAdam(self.shared_model.parameters(), lr=self.args.lr)
        self.optim.share_memory()  # Ensures shared optimizer is in shared memory

        self.vis = visdom.Visdom()
        self.main_update_step = Value('d', 0)

        if self.args.load_weight != 0:
            self.load_model(self.args.load_weight)

        self.jobs = []
        if self.args.t_flag:
            for process_id in range(self.args.jobs):
                job = A3CSingleProcess(process_id, self, logger_)
                self.jobs.append(job)

        self.test_win = None

    def train(self):
        test_p = mp.Process(target=self.test)
        self.jobs.append(test_p)
        self.args.train_step = 0
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()

    def test_sync(self):
        pass

    def test(self, render_=False):
        test_env = AtariEnv(gym.make(self.args.game), self.args.frame_seq, self.args.frame_skip, render=render_)
        test_model = A3CLSTMNet(self.env.state_shape, self.env.action_dim)

        while True:
            terminal = False
            reward_ = 0
            with torch.no_grad():
                lstm_h = torch.zeros(1, 256)
                lstm_c = torch.zeros(1, 256)
                test_env.reset_env()

                if int(self.main_update_step.value) % 500 == 0:
                    print("step:", int(self.main_update_step.value))
                    episode_length = 0
                    self.save_model(int(self.main_update_step.value))
                    test_model.load_state_dict(self.shared_model.state_dict())

                    while not terminal:
                        state_tensor = torch.from_numpy(test_env.state).float()
                        pl, v, (lstm_h, lstm_c) = test_model(state_tensor, (lstm_h, lstm_c))
                        action = pl.max(1)[1].item()
                        _, reward, terminal = test_env.forward_action(action)
                        reward_ += reward
                        episode_length += 1

                    print("Reward:", reward_)
                    print("Episode length:", episode_length)

    def save_model(self, name):
        path = os.path.join(self.args.train_dir, f"{name}_weight.pt")
        torch.save(self.shared_model.state_dict(), path)

    def load_model(self, name):
        path = os.path.join(self.args.train_dir, f"{name}_weight.pt")
        self.shared_model.load_state_dict(torch.load(path))


def loggerConfig():
    ts = str(time.strftime('%Y-%m-%d-%H-%M-%S'))
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s %(levelname)-2s %(message)s')

    fileHandler_ = logging.FileHandler(os.path.join(log_dir, f"a3c_training_log_{ts}.log"))
    fileHandler_.setFormatter(formatter)
    logger.addHandler(fileHandler_)
    logger.setLevel(logging.INFO)
    return logger


parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, default='ALE/Pong-v5')
parser.add_argument("--train_dir", type=str, default='./models/')
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--use_lstm", type=int, default=0)
parser.add_argument("--t_max", type=int, default=20)
parser.add_argument("--t_train", type=int, default=int(1e9))
parser.add_argument("--t_test", type=int, default=int(1e4))
parser.add_argument("--t_flag", type=int, default=1)
parser.add_argument("--jobs", type=int, default=16)
parser.add_argument("--frame_skip", type=int, default=1)
parser.add_argument("--frame_seq", type=int, default=1)
parser.add_argument("--opt", type=str, default="rms")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--grad_clip", type=float, default=40.0)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--entropy_beta", type=float, default=1e-5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--load_weight", type=int, default=0)

if __name__ == "__main__":
    args_ = parser.parse_args()
    logger = loggerConfig()
    model = A3CAtari(args_, logger)

    if args_.t_flag:
        print("====== Training ======")
        model.train()
    else:
        print("====== Testing ======")
        model.test(True)
