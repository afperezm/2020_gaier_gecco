import gym
import math
import numpy as np
import torch

from collections import OrderedDict
from domain.locomotion.networks import Actor, PolicyNet
from functools import partial
import pymap_elites.map_elites as map_elites

NUM_HIDDEN = 64
NUM_ACTIONS = 4
OBSERVATION_SIZE = 8


def random_layer_params(out_features, in_features):

    weight_param = torch.nn.parameter.Parameter(torch.zeros([out_features, in_features]), requires_grad=False)
    bias_param = torch.nn.parameter.Parameter(torch.zeros([out_features]), requires_grad=False)

    weight = torch.nn.init.kaiming_uniform_(weight_param, a=math.sqrt(5))

    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight_param)
    bound = 1 / math.sqrt(fan_in)

    bias = torch.nn.init.uniform_(bias_param, -bound, bound)

    return weight.flatten(), bias.flatten()


def random_params(dim_x, params):

    fc1_weight, fc1_bias = random_layer_params(NUM_HIDDEN, OBSERVATION_SIZE)
    fc2_weight, fc2_bias = random_layer_params(NUM_HIDDEN, NUM_HIDDEN)
    fc3_weight, fc3_bias = random_layer_params(NUM_ACTIONS, NUM_HIDDEN)

    x = torch.hstack([fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias]).numpy()

    assert dim_x == x.shape[0]

    return x


def evaluate_lander(x):

    # Get env info to initialize networks
    env = gym.make("LunarLander-v2")

    # Build state dictionary
    state_dict = OrderedDict()
    state_dict['fc1.0.weight'] = torch.tensor(x[0:NUM_HIDDEN * OBSERVATION_SIZE].reshape(NUM_HIDDEN, OBSERVATION_SIZE))
    state_dict['fc1.0.bias'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN])
    state_dict['fc2.0.weight'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN].reshape(NUM_HIDDEN, NUM_HIDDEN))
    state_dict['fc2.0.bias'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN])
    state_dict['fc3.0.weight'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_ACTIONS].reshape(NUM_ACTIONS, NUM_HIDDEN))
    state_dict['fc3.0.bias'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_ACTIONS:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_ACTIONS + NUM_ACTIONS])

    # Create new policy
    policy = PolicyNet(OBSERVATION_SIZE, NUM_ACTIONS, n_hidden=NUM_HIDDEN)

    # Load state dictionary
    policy.load_state_dict(state_dict)

    # Reset environment
    observation, info = env.reset()

    # Initialize counters
    total_reward = 0.0
    impact_x_pos = None
    impact_y_vel = None
    all_y_velocities = []

    # Evaluation loop
    while True:

        # Take action given current policy
        observation_tensor = torch.FloatTensor(observation)
        actions_probabilities = policy(observation_tensor)

        # Create action probability mass function
        actions_pmf = torch.distributions.Categorical(actions_probabilities)

        # Sample actions probability mass function
        action_tensor = actions_pmf.sample()
        action = action_tensor.item()

        # Take action on current state and observe new state and reward
        observation, reward, terminated, truncated, info = env.step(action)

        # Retrieve state statistics
        x_pos = observation[0]
        y_vel = observation[3]
        leg0_touch = bool(observation[6])
        leg1_touch = bool(observation[7])
        all_y_velocities.append(y_vel)

        # Update counters
        total_reward += reward

        # Check if the lunar lander is impacting for the first time
        if impact_x_pos is None and (leg0_touch or leg1_touch):
            impact_x_pos = x_pos
            impact_y_vel = y_vel

        # Break loop if game is done
        if terminated or truncated:
            break

    env.close()

    # If the lunar lander did not land, set the x-pos to the one from the final
    # time step, and set the y-vel to the max y-vel (we use min since the lander
    # goes down).
    if impact_x_pos is None:
        impact_x_pos = x_pos
        impact_y_vel = min(all_y_velocities)

    return total_reward, (impact_x_pos, impact_y_vel)


def evaluate(x):
    total_reward, behavior_descriptor = [], []
    for _ in range(1):
        result = evaluate_lander(x)
        total_reward.append(result[0])
        behavior_descriptor.append(result[1])
    return np.mean(total_reward), np.mean(behavior_descriptor, axis=0)


class LunarLander:
    def __init__(self, n_params):
        self.x_dims = n_params
        self.desc_length = 2

        # MAP-Elites Parameters
        params = map_elites.default_params
        params["min"] = [-1.] * self.x_dims
        params["max"] = [1.] * self.x_dims

        self.params = params

    def express(self, xx):
        return xx

    def randomInd(self, dim_x, params):
        return random_params(dim_x, params)
