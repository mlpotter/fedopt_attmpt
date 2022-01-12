# FedOPT attempt using PyTorch and `torch.distributed.rpc`

## client.py
* client.py contains the `client` object
* logs client actions and metrics to `client{rank}.log`

## server.py
* server.py contains the `server` object
* logs server actions and metrics to `server{rank}.log`

## data/mnist_flat/mnist_flat_generator.py
* generates non-iid dataloader for each client as `.pt` file. Loaded onto the `client` when initialized.
* Latent Dirichlet Allocation sampling from `FedML` repo

Run `python fedopt.py --epochs=2 --iterations=25 --world_size=5 --global_lr=0.01 --datapath=data/mnist_flat` in terminal to simulate FedAVG with 4 clients and 1 server for the mnist flattened dataset (or may try Iris).

## arguments for script
```
optional arguments:
  -h, --help            show this help message and exit
  --world_size WORLD_SIZE
                        The world size which is equal to 1 server + (world size - 1) clients
  --epochs EPOCHS       The number of epochs to run on the client training each iteration
  --iterations ITERATIONS
                        The number of iterations to communication between clients and server
  --batch_size BATCH_SIZE
                        The batch size during the epoch training
  --partition_alpha PARTITION_ALPHA
                        Number to describe the uniformity during sampling (heterogenous data generation for LDA)
  --datapath DATAPATH   folder path to all the local datasets
  --local_lr LOCAL_LR   Learning rate of local client (SGD)
  --global_lr GLOBAL_LR
                        Learning rate for global server (Adam)
  --client_percent CLIENT_PERCENT
                        Number of clients to sample for training and update
```

Will add more notes when ironed out , and understand `torch.distributed.rpc` better.