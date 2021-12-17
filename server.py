import torch.optim as optim
import torch.distributed.rpc as rpc
import logging
from client import client
import torch

class server(object):
    def __init__(self,
                 rank,
                 world_size,
                 model,
                 args):

        self.rank = rank
        self.world_size = world_size

        self.model = model
        self.optimizer = optim.Adam(model.parameters(),
                                   lr=args.global_lr)

        self.start_logger()
        self.init_clients(args)

        self.logger.info("Initialize Server")

    def init_clients(self,args):
        self.logger.info(f"Initialize {self.world_size-1} Clients")
        self.client_rrefs = [rpc.remote(to=f"client{rank}",
                                        func=client,
                                        args=(rank,self.world_size,self.model,args))
                             for rank in range(1,self.world_size)]

    def train(self):
        self.logger.info("Initializing Training")
        check_train = [client_rref.rpc_async(timeout=0).train() for client_rref in self.client_rrefs]
        [fut.wait() for fut in check_train]

    def evaluate(self):
        self.logger.info("Initializing Evaluation")
        total = []
        num_corr = []
        check_eval = [client_rref.rpc_async(timeout=0).evaluate() for client_rref in self.client_rrefs]
        for check in check_eval:
            corr, tot = check.wait()
            total.append(tot)
            num_corr.append(corr)

        self.logger.info("Accuracy over all data: {:.3f}".format(sum(num_corr)/sum(total)))

    def get_client_pseudo_grads(self):
        self.logger.info("Get Clients Pseudo Gradients")
        return [client_rref.rpc_async(timeout=0).send_pseudo_grad() for client_rref in self.client_rrefs]

    def get_client_sample_nums(self):
        self.logger.info("Get Clients Sample Numbers")
        return [client_rref.rpc_async(timeout=0).send_num_train() for client_rref in self.client_rrefs]

    def aggregate(self):
        pseudo_grads = self.get_client_pseudo_grads()
        client_num_train = self.get_client_sample_nums()

        pseudo_grads = [fut.wait() for fut in pseudo_grads]
        client_num_train = [fut.wait() for fut in client_num_train]
        total_train = sum(client_num_train)

        self.optimizer.zero_grad()

        # probably need to reavluate this...
        for (param_name,param) in self.model.state_dict().items():
            self.model.get_parameter(param_name).grad = torch.zeros_like(param)

            for n_train,pseudo_grad in zip(client_num_train,pseudo_grads):
                self.model.get_parameter(param_name).grad = self.model.get_parameter(param_name).grad + (n_train / total_train) * pseudo_grad[param_name]

        self.optimizer.step()


    def send_global_weights(self):
        self.logger.info(f"Send Global Weights to {self.world_size-1} Clients")
        global_weights = self.model.state_dict()
        check_loaded = [client_rref.rpc_async(timeout=0).get_global_weights(global_weights) for client_rref in self.client_rrefs]
        [check.wait() for check in check_loaded]

    def start_logger(self):
        self.logger = logging.getLogger(f"client{self.rank}")
        self.logger.setLevel(logging.INFO)

        format = logging.Formatter("%(asctime)s: %(message)s")

        fh = logging.FileHandler(filename=f"logs/server{self.rank}.log",mode='w')
        fh.setFormatter(format)
        fh.setLevel(logging.INFO)

        self.logger.addHandler(fh)
