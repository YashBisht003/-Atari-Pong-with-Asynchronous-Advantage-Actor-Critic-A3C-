

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from environment import AtariEnv
import torch.multiprocessing as mp
import numpy as np
import gymnasium as gym

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True)).expand_as(out)
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class A3CLSTMNet(nn.Module):

    def __init__(self, state_shape, action_dim,):
        super(A3CLSTMNet, self).__init__()
        self.state_shape = state_shape
        self.action_dim = action_dim

        self.conv1 = nn.Conv2d(self.state_shape[0], 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(3 * 3 * 32, 256, 1)
        # policy output
        self.linear_policy_1 = nn.Linear(256, self.action_dim)
        self.softmax_policy = nn.Softmax(dim=1)
        # value output
        self.linear_value_1 = nn.Linear(256, 1)

        self.apply(weights_init)
        self.linear_policy_1.weight.data = normalized_columns_initializer(self.linear_policy_1.weight.data, 0.01)
        self.linear_policy_1.bias.data.fill_(0)
        self.linear_value_1.weight.data = normalized_columns_initializer(self.linear_value_1.weight.data, 1.0)
        self.linear_value_1.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x, hidden):
        # Reshape input to [batch, channels, height, width]
        x = x.view(-1, self.state_shape[0], self.state_shape[1], self.state_shape[2]).to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 3 * 3 * 32)
        x, c = self.lstm(x, (hidden[0], hidden[1]))
        pl = self.linear_policy_1(x)
        pl = self.softmax_policy(pl)
        v = self.linear_value_1(x)
        return pl, v, (x, c)


class A3CSingleProcess(mp.Process):

    def __init__(self, process_id, master, logger_):
        super(A3CSingleProcess, self).__init__(name="process_%d" % process_id)
        self.process_id = process_id
        self.master = master
        self.logger = logger_
        #self.device = device  # Device passed to constructor
        self.args = master.args
        self.env = AtariEnv(gym.make(self.args.game), self.args.frame_seq, self.args.frame_skip)  # Using gymnasium
        self.local_model = A3CLSTMNet(self.env.state_shape, self.env.action_dim)  # Pass device here
        # sync the weights at the beginning
        self.sync_network()
        self.loss_history = []
        self.win = None
        self.state_final = None
        self.Image = None
    def sync_network(self):
        self.local_model.load_state_dict(self.master.shared_model.state_dict())

    def forward_explore(self, hidden):
        terminal = False
        t_start = 0
        rollout_path = {"state": [], "action": [], "rewards": [], "done": []}
        pl_roll = []
        v_roll = []
        while not terminal and (t_start <= self.args.t_max):
            t_start += 1
            state_ = self.env.state
            state_tensor = torch.tensor(state_, dtype=torch.float32).to(self.device)
            pl, v, hidden = self.local_model(state_tensor, hidden)
            pl_roll.append(pl)
            v_roll.append(v)

            # Sample an action; ensure num_samples=1 and then extract as an integer
            action_tensor = pl.multinomial(num_samples=1)
            action = action_tensor.item()
            self.state_final, reward, terminal = self.env.forward_action(action)

            rollout_path["state"].append(state_)
            rollout_path["action"].append(action)
            rollout_path["rewards"].append(reward)
            rollout_path["done"].append(terminal)

        return rollout_path, hidden, pl_roll, v_roll

    def discount(self, rewards):
        discounted_rewards = torch.zeros_like(rewards).to(self.device)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + self.args.gamma * running_add
            discounted_rewards[t] = running_add
        return discounted_rewards

    def run(self):
        self.env.reset_env()
        loop = 0
        lstm_h = torch.zeros(1, 256).to(self.device)
        lstm_c = torch.zeros(1, 256).to(self.device)
        while True:
            loop += 1

            rollout_path, (lstm_h, lstm_c), p_roll, v_roll = self.forward_explore((lstm_h, lstm_c))
            if rollout_path["done"][-1]:
                rollout_path["rewards"].append(0)
                self.env.reset_env()
                lstm_h = torch.zeros(1, 256).to(self.device)
                lstm_c = torch.zeros(1, 256).to(self.device)
            else:
                state_tensor = torch.tensor(self.state_final, dtype=torch.float32).to(self.device)
                _, v_t, _ = self.local_model(state_tensor, (lstm_h, lstm_c))
                lstm_h = lstm_h.detach()
                lstm_c = lstm_c.detach()
                rollout_path["rewards"].append(v_t.item())

            # Calculate returns
            rollout_path["returns"] = self.discount(rollout_path["rewards"])

            loss = self.PathBackProp(rollout_path, p_roll, v_roll)
            self.loss_visual(loss, loop)
            self.master.main_update_step.value += 1
            self.sync_network()

    def loss_visual(self, loss_, loop_):
        self.loss_history.append(loss_)
        if loop_ > 2:
            Y_ = np.array(self.loss_history).reshape(-1, 1)
            self.win = self.master.vis.line(Y=Y_, X=np.arange(len(self.loss_history)), win=self.win)
            # Optional: Visualize image state
            # self.Image = self.master.vis.image(np.resize(self.state_final, (160,160)), win=self.Image)

    def ensure_shared_grads(self, model, shared_model):
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
            shared_param.grad.data.copy_(param.grad.data)

    def PathBackProp(self, rollout_path_, p_roll, v_roll):
        state = torch.tensor(np.array(rollout_path_['state']), dtype=torch.float32).to(self.device)
        target_q = torch.tensor(rollout_path_['returns'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(rollout_path_['action'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(rollout_path_['rewards'], dtype=torch.float32).to(self.device)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1).to(self.device)
        for i in reversed(range(len(p_roll))):
            log_prob = torch.log(p_roll[i])
            entropy = -torch.sum(log_prob * p_roll[i], dim=1).mean()
            action_tensor = actions[i].view(1, 1)
            log_prob_ = log_prob.gather(1, action_tensor)
            advantage = target_q[i] - v_roll[i]
            value_loss += 0.5 * advantage.pow(2)
            if i != (len(p_roll) - 1):
                delta_t = rewards[i] + self.args.gamma * v_roll[i + 1].item() - v_roll[i].item()
            else:
                delta_t = rewards[i] + self.args.gamma * rewards[i + 1] - v_roll[i].item()

            gae = gae * self.args.gamma + delta_t
            policy_loss = policy_loss - log_prob_ * gae - 0.01 * entropy

        self.master.optim.zero_grad()
        loss_all = 0.5 * value_loss + policy_loss
        loss_all.backward()
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 40)
        self.ensure_shared_grads(self.local_model, self.master.shared_model)
        self.master.optim.step()
        self.logger.info("pl_loss %f, v_loss %f", policy_loss.item(), value_loss.item())
        return loss_all.item()
