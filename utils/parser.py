import argparse

__all__ = ['set_parser']

def set_parser():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu_id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100','SVHN','fmnist','tinyimage'],
                        help='dataset name')
    parser.add_argument('--arch', default='wideresnet', type=str,
                        # choices=['wideresnet', 'resnext'],
                        help='model name')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--wdecay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--seed', default=5, type=int,
                        help="random seed")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    #my
    parser.add_argument("--warm_up", type=int, default=10,
                        help="epoch to warmup")
    parser.add_argument("--partial_rate", type=float, default=0.1,
                        help="epoch to warmup")
    parser.add_argument("--k", type=int, default=3,
                        help="time line")
    parser.add_argument("--epochs", type=int, default=200,
                        help="time line")
    parser.add_argument('--select_threshold', default=0.9, type=float,
                        help='select threshold')                    
    parser.add_argument('--lambda_cr', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--data_path', default='/home/sytian/data/',
                        help='directory where puts data')
    parser.add_argument('--use_hierarchical', action='store_true',
                        help="cifar 100 label")
    parser.add_argument('--use_mix', action='store_true',
                        help="wheather to use mix up")
    parser.add_argument('--sharpen_T', default=1, type=float,
                        help='temprature of sharpen')
    parser.add_argument('--alpha', default=0.75, type=float,
                        help='temprature of sharpen')        
    # /home/featurize/data'
    # '/data/tsy/data/'
    # '/home/tsy/data/'
    # /home/sytian/data/

    args = parser.parse_args()
    return args
    