import gym
import math
import numpy as np
import QDgym
import torch

from collections import OrderedDict
from domain.locomotion.networks import Actor
from functools import partial
import pymap_elites.map_elites as map_elites


def random_layer_params(out_features, in_features):
    weight_param = torch.nn.parameter.Parameter(torch.zeros([out_features, in_features]), requires_grad=False)
    bias_param = torch.nn.parameter.Parameter(torch.zeros([out_features]), requires_grad=False)
    #
    weight = torch.nn.init.xavier_uniform_(weight_param)
    #
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight_param)
    bound = 1 / math.sqrt(fan_in)
    bias = torch.nn.init.uniform_(bias_param, -bound, bound)
    #
    return weight.flatten(), bias.flatten()


def random_params(dim_x, params):
    #
    l1_weight, l1_bias = random_layer_params(128, 28)
    l2_weight, l2_bias = random_layer_params(128, 128)
    l3_weight, l3_bias = random_layer_params(8, 128)
    #
    x = torch.hstack([l1_weight, l1_bias, l2_weight, l2_bias, l3_weight, l3_bias]).numpy()
    #
    assert dim_x == x.shape[0]
    #
    return x


def evaluate(x, env_id):

    # Get env info to initialize networks
    env = gym.make(env_id)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Function that creates new actor
    actor_fn = partial(Actor,
                       state_dim,
                       action_dim,
                       max_action,
                       neurons_list=[128, 128],
                       normalise=False,
                       affine=False)

    # Create new actor
    actor = actor_fn()

    # Build state dictionary
    state_dict = OrderedDict()
    state_dict['l1.weight'] = torch.tensor(x[0:128 * state_dim].reshape(128, state_dim))
    state_dict['l1.bias'] = torch.tensor(x[128 * state_dim:128 * state_dim + 128])
    state_dict['l2.weight'] = torch.tensor(x[128 * state_dim + 128:128 * state_dim + 128 + 128 * 128].reshape(128, 128))
    state_dict['l2.bias'] = torch.tensor(x[128 * state_dim + 128 + 128 * 128:128 * state_dim + 128 + 128 * 128 + 128])
    state_dict['l3.weight'] = torch.tensor(x[128 * state_dim + 128 + 128 * 128 + 128:128 * state_dim + 128 + 128 * 128 + 128 + 128 * action_dim].reshape(action_dim, 128))
    state_dict['l3.bias'] = torch.tensor(x[128 * state_dim + 128 + 128 * 128 + 128 + 128 * action_dim:128 * state_dim + 128 + 128 * 128 + 128 + 128 * action_dim + action_dim])

    # Load state dictionary
    actor.load_state_dict(state_dict)

    # # start environment simulation
    # env = gym.make("QDWalker2DBulletEnv-v0")

    # get a new actor to evaluate
    # env.seed(int((master_seed + 100) * evaluation_id))
    state = env.reset()
    done = False

    # eval loop
    while not done:
        state = torch.FloatTensor(state.reshape(1, -1))
        action = actor(state).cpu().data.numpy().flatten()
        next_state, reward, done, _ = env.step(action)
        # # env_screen = np.expand_dims(env.render(mode='rgb_array'), axis=0)
        # done_bool = float(done) if env.T < env._max_episode_steps else 0
        # if env.T == 1:
        #     state_array = state
        #     action_array = action
        #     next_state_array = next_state
        #     reward_array = reward
        #     done_bool_array = done_bool
        #     # env_screen_array = env_screen
        # else:
        #     state_array = np.vstack((state, state_array))
        #     action_array = np.vstack((action, action_array))
        #     next_state_array = np.vstack((next_state, next_state_array))
        #     reward_array = np.vstack((reward, reward_array))
        #     done_bool_array = np.vstack((done_bool, done_bool_array))
        #     # env_screen_array = np.vstack((env_screen, env_screen_array))
        state = next_state

    # if eval_mode:
    #     return actor_idx, (state_array, action_array, next_state_array, reward_array, done_bool_array)
    # else:

    env.close()

    return env.tot_reward, env.desc


def evaluate_ant(x):
    # total_reward, behavior_descriptor = [], []
    # for _ in range(10):
    #     result = evaluate(x, "QDAntBulletEnv-v0")
    #     total_reward.append(result[0])
    #     behavior_descriptor.append(result[1])
    # return np.mean(total_reward), np.mean(behavior_descriptor, axis=0)
    return evaluate(x, "QDAntBulletEnv-v0")


class Ant:
    def __init__(self, n_params):
        self.x_dims = n_params
        self.desc_length = 4

        # MAP-Elites Parameters
        self.params = map_elites.default_params

        self.params["min"] = [-1.] * self.x_dims
        self.params["max"] = [1.] * self.x_dims

    def express(self, xx):
        return xx

    def randomInd(self, dim_x, params):
        return random_params(dim_x, params)
