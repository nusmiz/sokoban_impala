import torch
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from pathlib import Path

import models


def predict_func(states):
    states = torch.from_numpy(states).to(device)
    model.eval()
    with torch.no_grad():
        pi = model.pi(states)
        probs = F.softmax(pi, dim=1)
        actions = probs.multinomial(1)
        policy = probs.gather(1, actions)
        actions = np.squeeze(actions.cpu().numpy(), axis=1)
        policy = np.squeeze(policy.cpu().numpy(), axis=1)
        return actions, policy


def calc_vs_and_pg_advantages(states, actions, rewards, behaviour_policy, data_sizes):
    t_max = len(data_sizes) - 1
    model.eval()
    vs_list = []
    pg_advantage_list = []
    with torch.no_grad():
        prev_data_size = data_sizes[-1]
        prev_value = model.v(states[t_max][:prev_data_size])
        prev_v = prev_value
        sum_delta = 0
        for i in reversed(range(0, t_max)):
            data_size = data_sizes[i]
            if prev_data_size != data_size:
                prev_value = torch.cat((prev_value, torch.zeros(
                    data_size - prev_data_size, 1).to(device)), 0)
                prev_v = torch.cat((prev_v, torch.zeros(
                    data_size - prev_data_size, 1).to(device)), 0)
            pi, value = model.forward(states[i][:data_size])
            probs = F.softmax(pi, dim=1)
            target_policy = probs.gather(1, actions[i][:data_size])
            rho = target_policy / behaviour_policy[i][:data_size]
            rho = torch.min(rho, clip_rho_threshold)
            delta = rho * (rewards[i][:data_size] + gamma * prev_value - value)
            sum_delta = delta + gamma * rho * sum_delta
            vs = sum_delta + value
            pg_advantage = rho * (rewards[i][:data_size] + gamma * prev_v - value)
            vs_list = [vs.detach()] + vs_list
            pg_advantage_list = [pg_advantage.detach()] + pg_advantage_list
            prev_data_size = data_size
            prev_value = value
            prev_v = vs
        return vs_list, pg_advantage_list


def calc_loss(states, actions, vs, pg_advantages, data_sizes):
    num_of_data = sum(data_sizes)
    v_loss = 0
    pi_loss = 0
    entropy_loss = 0
    for i in range(0, len(data_sizes)):
        data_size = data_sizes[i]
        pi, v = model.forward(states[i][:data_size])
        v_loss += 0.5 * (v - vs[i]).pow(2).sum()
        probs = F.softmax(pi, dim=1)
        pi_loss += -(torch.max(F.log_softmax(pi, 1).gather(1, actions[i][:data_size]),
                               log_epsilon) * pg_advantages[i]).sum()
        entropy_loss += (torch.max(F.log_softmax(pi, 1), log_epsilon) * probs).sum(1).sum()
    return v_loss / num_of_data, pi_loss / num_of_data, entropy_loss / num_of_data


def train_func(states, actions, rewards, behaviour_policy, data_sizes):
    states = torch.from_numpy(states.reshape(len(data_sizes), -1, 3, 80, 80)).to(device)
    actions = torch.from_numpy(actions).to(device)
    rewards = torch.from_numpy(rewards).to(device)
    behaviour_policy = torch.from_numpy(behaviour_policy).to(device)
    vs, pg_advantages = calc_vs_and_pg_advantages(
        states, actions, rewards, behaviour_policy, data_sizes)
    model.train()
    optimizer.zero_grad()
    v_loss, pi_loss, entropy_loss = calc_loss(states, actions, vs, pg_advantages, data_sizes[:-1])
    loss = (0.5 * v_loss + pi_loss + beta * entropy_loss)
    loss.backward()
    optimizer.step()
    return v_loss.item(), pi_loss.item(), entropy_loss.item()


def save_model(index):
    output_dir = Path(f"output/{index}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pth")
    torch.save(optimizer.state_dict(), output_dir / "optimizer.pth")


def load_model(index):
    model_dir = Path(f"output/{index}").resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(torch.load(model_dir / "model.pth"))
    optimizer.load_state_dict(torch.load(model_dir / "optimizer.pth"))


device = torch.device("cpu")
model = models.A3CModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.003)

gamma = 0.99
log_epsilon = torch.Tensor([math.log(1e-6)]).to(device)
clip_rho_threshold = torch.Tensor([math.log(1.0)]).to(device)
beta = 1e-3