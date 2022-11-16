import argparse
import json
import numpy as np
import os
import pymap_elites.map_elites as map_elites
import time
import torch

from vae import VecVAE
from vae_map_elites import vae_map_elites

PARAMS = None


def main():
    # -- Parse input arguments ----------------------------------------------------#
    # python3 dde.py arm20  map 1500                   # map-elites w/new cvt-archive
    # python3 dde.py arm20  map maps/ring_6.dat        # map-elites w/predefined archive
    # python3 dde.py arm20  vae maps/ring_6.dat 10     # dde-elites w/10 latent dimensions
    # python3 dde.py arm200 vae maps/ring_6.dat 32     # dde-elites w/32 latent dimensions
    # python3 dde.py arm20  dde maps/ring_6.dat dde/dde_arm20.pt # map-elites dde representation

    # domain = sys.argv[1]  # arm | hex
    domain = PARAMS.domain
    # mode = sys.argv[2]  # map | vae | dde
    mode = PARAMS.mode
    # archive = sys.argv[3]  # centroid_file | n_centroids
    num_niches = PARAMS.num_niches
    centroids_file = PARAMS.centroids_file
    latent_length = PARAMS.latent_length
    vae_file = PARAMS.vae_file
    n_gen = PARAMS.num_generations
    sigma_iso = PARAMS.sigma_iso
    sigma_line = PARAMS.sigma_line
    batch_size = PARAMS.batch_size
    random_init = PARAMS.random_init
    random_init_batch = PARAMS.random_init_batch
    save_freq = PARAMS.save_freq
    print_freq = PARAMS.print_freq

    exp_name = f"{mode}_{domain}_{time.strftime('%y%m%d')}-{time.strftime('%H%M%S')}"

    # Create experiment directory
    if not os.path.exists(os.path.join("results", exp_name)):
        os.makedirs(os.path.join("results", exp_name))

    # Dump program arguments
    with open(os.path.join("results", exp_name, "params.json"), "w") as f:
        json.dump(vars(PARAMS), f)

    if mode == 'map':
        print('\n[****] Running MAP-Elites [****]')
    elif mode == 'line':
        print('\n[****] Running MAP-Elites w/Line Search [****]')
    elif mode == 'vae_only':
        print('\n[****] Running VAE-Elites using only VAE [****]')
        # nZ = int(sys.argv[4])  # number of latent dimensions
    elif mode == 'vae':
        print('\n[****] Running VAE-Elites [****]')
        # nZ = int(sys.argv[4])  # number of latent dimensions
    elif mode == 'vae_line':
        print('\n[****] Running VAE-Elites w/line mutation[****]')
        # nZ = int(sys.argv[4])  # number of latent dimensions
    elif mode == 'dde':
        print('\n[****] Running DDE-Elites [****]')
        # vaeFile = sys.argv[4]  # .pt file of pretrained VAE
    else:
        print('Invalid mode selected (map/vae/dde)')
        exit(1)

    # -- Setup --------------------------------------------------------------------#
    # Load or Create Archive (number of niches or pre-defined centroids)
    if centroids_file:  # Centroid file
        centroids = np.loadtxt(centroids_file)  # centroids = np.loadtxt('maps/ring_6.dat')
        num_niches = np.shape(centroids)[0]
    else:
        centroids = np.empty(shape=(0, 0))

    # Set Domain
    if domain[0:3] == 'arm':
        from domain.arm.planarArm import Arm2d, Arm, evaluate
        n_joints = int(domain[3:])
        print("Number of Joints", n_joints)
        d = Arm(n_joints)  # Numbers after arm is number of joints
        # if n_joints > 200:
        #     d.params["random_init"] = 0.01
        # else:
        #     d.params["random_init"] = 0.05
        # n_gen = 10000
    elif domain == 'hex':
        from domain.hexa.hexapod import Hex, evaluate
        d = Hex()
        # d.params["random_init"] = 0.05
        # n_gen = 50000
    elif domain == 'ant':
        from domain.locomotion.ant import Ant, evaluate_ant as evaluate
        d = Ant(21256)
        # n_gen = 10000
    elif domain == 'walker':
        from domain.locomotion.walker import Walker, evaluate_walker as evaluate
        d = Walker(20230)
        # n_gen = 100000
    elif domain == 'lander':
        from domain.gym.lander import LunarLander, evaluate
        d = LunarLander(4996)
    else:
        print('Invalid Domain (e.g. hex/arm20/arm200/walker/ant)')
        exit(1)

    x_dims = d.x_dims
    desc_length = d.desc_length
    params = d.params
    params["gen_to_phen"] = d.express
    params["random"] = d.randomInd
    params["batch_size"] = batch_size
    params["random_init"] = random_init
    params["random_init_batch"] = random_init_batch
    params["trainMod"] = 1  # number of gens between VAE training
    params["trainEpoch"] = 5
    params["banditWindow"] = 1000
    params["vector_variation"] = False

    n_gen += 1

    # Set Logging Hyperparameters
    params["dump_period"] = save_freq
    params["save_format"] = 'bin'
    params["print_mod"] = print_freq
    map_elites_log_path = os.path.join('results', exp_name, 'map_elites_log.dat')
    vae_log_path = os.path.join('results', exp_name, 'vae_log.dat')

    # Tiny run testing
    # params["random_init"] = 5./float(n_niches)
    # n_gen = 3

    # -- Test Algorithms ----------------------------------------------------------#
    if mode == 'map':
        params["sigma_iso"] = sigma_iso
        params["sigma_line"] = 0.0
        # params["vector_variation"] = False
        map_elites.compute(desc_length, x_dims, evaluate, params=params,
                           centroids=centroids, n_niches=num_niches, n_gen=n_gen,
                           log_filepath=map_elites_log_path)

    if mode == 'line':
        params["sigma_iso"] = sigma_iso
        params["sigma_line"] = sigma_line
        # params["vector_variation"] = False
        map_elites.compute(desc_length, x_dims, evaluate, params=params,
                           centroids=centroids, n_niches=num_niches, n_gen=n_gen,
                           log_filepath=map_elites_log_path)

    if mode == 'vae':
        params["bandit_prob_xover"] = [0, 0.25, 0.5, .75, 1.0]
        params["bandit_line_sigma"] = [0.0]
        vae_map_elites(desc_length, x_dims, evaluate, params,
                       centroids=centroids, n_niches=num_niches, n_gen=n_gen,
                       model=VecVAE, latent_length=latent_length, vae_log_filepath=vae_log_path,
                       log_filepath=map_elites_log_path)

    if mode == 'vae_line':
        params["bandit_prob_xover"] = [0, 0.25, 0.5, .75, 1.0]
        # params["bandit_prob_xover"] = [0, 1.0]
        params["bandit_line_sigma"] = [0.1, 0.0]

        vae_map_elites(desc_length, x_dims, evaluate, params,
                       centroids=centroids, n_niches=num_niches, n_gen=n_gen,
                       model=VecVAE, latent_length=latent_length, vae_log_filepath=vae_log_path,
                       log_filepath=map_elites_log_path)

    if mode == 'vae_only':
        params["bandit_prob_xover"] = [1.0]
        params["bandit_line_sigma"] = [0.0]  # this is never used
        vae_map_elites(desc_length, x_dims, evaluate, params,
                       centroids=centroids, n_niches=num_niches, n_gen=n_gen,
                       model=VecVAE, latent_length=latent_length, vae_log_filepath=vae_log_path,
                       log_filepath=map_elites_log_path)

    # -- Use Data-Driven Encoding -------------------------------------------------#
    if mode == 'dde':
        print('\n[**] Loading Data Driven Encoding from: ', vae_file, ' [**]')
        weights = torch.load(vae_file)
        z_dims = list(weights()['fc21.bias'].size())[0]  # Decoder input
        x_dims = list(weights()['fc4.bias'].size())[0]  # Decoder output
        print('\n[*] Using ', str(x_dims), 'D Encoding with ', str(z_dims), ' Latent Dimensions [*]')

        n_gen = int(n_gen / 10) + 1
        params["dump_period"] = 100

        # Mutation Parameters
        params["min"] = [-5.0] * z_dims
        params["max"] = [5.0] * z_dims
        params['sigma_line'] = 0.1  # was 0.0 in original experiments...
        params["sigma_iso"] = 0.015
        params["random"] = d.random_vae_ind
        # params["vector_variation"] = False

        # Load DDE
        vae = VecVAE(x_dims, z_dims)
        vae.load_state_dict(weights())
        params["gen_to_phen"] = vae.express

        print('\n[**] Optimizing DDE with MAP-Elites [**]')
        map_elites.compute(desc_length, z_dims, evaluate, params=params,
                           centroids=centroids, n_niches=num_niches, n_gen=n_gen,
                           log_filepath=map_elites_log_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="Environment domain", required=True)
    parser.add_argument("--mode", help="Training algorithm", required=True)
    parser.add_argument("--num_niches", help="Number of niches", default=1500, type=int)
    parser.add_argument("--centroids_file", help="Centroids .dat file")
    parser.add_argument("--latent_length", help="Number of latent dimensions", type=int)
    parser.add_argument("--vae_file", help="Pretrained VAE file")
    parser.add_argument("--num_generations", help="Max number of generations", default=1000, type=int)
    parser.add_argument("--sigma_iso", help="Isometric mutation rate", default=0.01, type=float)
    parser.add_argument("--sigma_line", help="Line mutation rate", default=0.01, type=float)
    parser.add_argument("--batch_size", help="Batch for random mutation", default=100, type=int)
    parser.add_argument("--random_init", help="Proportion of niches to be filled before starting", default=0.1, type=float)
    parser.add_argument("--random_init_batch", help="Batch for random initialization", default=100, type=int)
    parser.add_argument("--save_freq", help="Frequency to dump archive", default=100, type=int)
    parser.add_argument("--print_freq", help="Frequency to print results", default=1, type=int)
    # parser.add_argument("--max_episodes", default=5000, help="Max number of episodes to play", type=int)
    # parser.add_argument("--gamma", default=0.99, help="Discount factor", type=float)
    # parser.add_argument("--policy_lr", default=0.001, help="Learning rate for policy network", type=float)
    # parser.add_argument("--base_lr", default=0.001, help="Learning rate for baseline network", type=float)
    # parser.add_argument("--policy_dims", default=64, help="Hidden dims of policy network", type=int)
    # parser.add_argument("--base_dims", default=64, help="Hidden dims of baseline network", type=int)
    # parser.add_argument("--policy_checkpoint", help="Policy network checkpoint file")
    # parser.add_argument("--base_checkpoint", help="Base network checkpoint file")
    # parser.add_argument('--render_frequency', type=int, default=0, metavar='N', help='Episode rendering frequency')
    # parser.add_argument("--checkpoint_dir", help="Checkpoints directory", required=True)
    # parser.add_argument("--summary_dir", help="Summary directory", required=True)
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
