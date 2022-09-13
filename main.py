# from baselines import logger
import argparse
import utils_storage as storage
import sys
from rapid.train import train

def argparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of RAPID")
    parser.add_argument('--env', help='environment ID', type=str, default='MiniGrid-ObstructedMaze-2Dlh-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--frames', help='Number of timesteps', type=int, default=int(4e7))
    parser.add_argument('--nsteps', help='nsteps', type=int, default=128)
    parser.add_argument('--log_dir', help='the directory to save log file', type=str, default='log')
    parser.add_argument('--lr', help='the learning rate', type=float, default=1e-4)
    parser.add_argument('--w0', help='weight for extrinsic rewards', type=float, default=1.0)
    parser.add_argument('--w1', help='weight for local bonus', type=float, default=0.1)
    parser.add_argument('--w2', help='weight for global bonus', type=float, default=0.001)
    parser.add_argument('--buffer_size', help='the size of the ranking buffer', type=int, default=10000)
    parser.add_argument('--batch_size', help='the batch size', type=int, default=256)
    parser.add_argument('--sl_until', help='SL until which timestep', type=int, default=100000000)
    parser.add_argument('--disable_rapid', help='Disable SL, i.e., PPO', action='store_true')
    parser.add_argument('--sl_num', help='Number of updated steps of SL', type=int, default=5)
    parser.add_argument('--ent_coef', help='Entropy coefficient value', type=float, default=0.01)

    # Define Self-Imitation-Learning (instead of RAPID)
    parser.add_argument('--use_sil', help='Self-Imitation-Learning', type=int, default=0)
    parser.add_argument('--sil_loss_weight', help='Weight to be applied to the loss', type=float, default=0.1)

    # Intrinsic motivation
    parser.add_argument('--im_coef', help='Intrinsic coefficient (0=no IM) ', type=float, default=0)
    parser.add_argument('--im_type', help='Intrinsic Module', type=str, default='counts')
    parser.add_argument('--use_ep_counts', help='Use episodic counts as regularizer for IM', type=int, default=0)
    parser.add_argument('--use_1st_counts', help='Use just 1st visit rewards in an episode for IM', type=int, default=0)

    # Network architecture
    parser.add_argument('--use_sharednetwork', help='Select if use a shared-network AC', type=int, default=0)

    # Gpus
    parser.add_argument('--gpu_id', help='Used to limit the capabilities of the simulation to a single gpu-card', type=int, default=-1)
    return parser

def main():
    parser = argparser()
    args = parser.parse_args()
    # logger.configure(dir=args.log_dir)

    # ******************************************************************************
    # Load loggers and Tensorboard writer
    # ******************************************************************************
    model_dir = storage.get_model_dir(args.log_dir)
    txt_logger = storage.get_txt_logger(model_dir)
    csv_file, csv_logger = storage.get_csv_logger(model_dir)
    # tb_writer = tensorboardX.SummaryWriter(model_dir)

    # log commands
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # define optimal score
    if args.env == 'MiniGrid-MultiRoom-N7-S4-v0':
        optimal_score = 0.775
    elif args.env == 'MiniGrid-MultiRoom-N10-S4-v0':
        optimal_score = 0.76
    elif args.env == 'MiniGrid-MultiRoom-N7-S8-v0':
        optimal_score = 0.65
    elif args.env == 'MiniGrid-KeyCorridorS3R3-v0':
        optimal_score = 0.9
    elif args.env == 'MiniGrid-KeyCorridorS4R3-v0':
        optimal_score = 0.9
    elif args.env == 'MiniGrid-ObstructedMaze-2Dlh-v0':
        optimal_score = 0.95
    else:
        optimal_score = 1
    # **************************************************************************
    # TRAIN
    # **************************************************************************
    train(args, args.gpu_id, txt_logger,(csv_file,csv_logger),optimal_score)

if __name__ == '__main__':
    main()
