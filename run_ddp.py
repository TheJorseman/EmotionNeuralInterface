import torch.multiprocessing as mp
from argparse import ArgumentParser
import os
import psutil
from workbench import Workbench
import yaml

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--config_yaml', default="config/config.yaml", type=str)
    return parser.parse_args()

def train(gpu, args):
    exp = Workbench(args.config_yaml, ddp=True)
    exp.run()

if __name__ == '__main__':
    args = arg_parser()
    data = {}
    with open(args.config_yaml) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    data = data['ddp']
    args.world_size = data['n_gpus'] * data['nodes']
    #if args.iface == 'auto':
    #    iface = list(filter(lambda x: 'en' in x, psutil.net_if_addrs().keys()))[0]
    #os.environ['GLOO_SOCKET_IFNAME'] = iface
    os.environ['MASTER_ADDR'] = data['ip']
    print("ip_adress is", data['ip'])
    os.environ['MASTER_PORT'] = data['master_port']
    os.environ['WORLD_SIZE'] = str(args.world_size)
    # nprocs: number of process which is equal to args.ngpu here
    mp.spawn(train, nprocs=data['n_gpus'], args=(args,))
    