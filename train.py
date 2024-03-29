import argparse
import shutil
from utils.learning.train_part import train
from pathlib import Path

import os, sys

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=8e-4, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=20, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_Unet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, nargs = '+', default=['C:\FastMRI/Data/train/image/', 'C:\FastMRI/Data/train/reconstructions_train/'], help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, nargs = '+', default=['C:\FastMRI/Data/val/image/', 'C:\FastMRI/Data/val/reconstructions_val/'], help='Directory of validation data')
        
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument('--input-key', type=str, nargs='+', default=['image_input','image_grappa'], help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name


    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)
