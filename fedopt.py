from server import server
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from model import *
import os
import argparse
from data.mnist_flat.mnist_flat_generator import *

def init_env():
    print("Initialize Meetup Spot")
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ["MASTER_PORT"] = "5689"

def example(rank,world_size,args):
    init_env()
    if rank == 0:
        rpc.init_rpc(f"server{rank}", rank=rank, world_size=world_size)
        Server = server(rank,
                        world_size,
                        TwoLayerNeuralNet(784, 392,10),
                        args)

        Server.send_global_weights()

        for iter in range(args.iterations):
            Server.train()
            Server.aggregate()
            Server.send_global_weights()
            Server.evaluate()
        rpc.shutdown()
    else:
        rpc.init_rpc(f"client{rank}", rank=rank, world_size=world_size)
        print("Client")
        rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedAVG Initialization')
    parser.add_argument('--world_size',type=int,default=3,help='The world size which is equal to 1 server + (world size - 1) clients')
    parser.add_argument('--epochs',type=int,default=3,help='The number of epochs to run on the client training each iteration')
    parser.add_argument('--iterations',type=int,default=1000,help='The number of iterations to communication between clients and server')
    parser.add_argument('--batch_size',type=int,default=16,help='The batch size during the epoch training')
    parser.add_argument('--partition_alpha',type=float,default=0.5,help='Number to describe the uniformity during sampling (heterogenous data generation for LDA)')
    parser.add_argument('--datapath',type=str,default="data/",help='folder path to all the local datasets')
    parser.add_argument('--local_lr',type=float,default=0.001,help='Learning rate of local client (SGD)')
    parser.add_argument('--global_lr',type=float,default=0.01,help='Learning rate for global server (Adam)')

    args = parser.parse_args()

    args.client_num_in_total = args.world_size - 1

    load_mnist_flat(args)

    world_size = args.world_size
    mp.spawn(example,
             args=(world_size,args),
             nprocs=world_size,
             join=True)