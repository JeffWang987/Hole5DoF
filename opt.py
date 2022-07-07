import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, default='./data',
                        help='root directory of dataset')
    parser.add_argument('--resize_fac', type=int, default=2,
                        help='input image resize factor')
    parser.add_argument('--topk_p', type=int, default=10,
                        help='number of predictions')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers for data loader')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()