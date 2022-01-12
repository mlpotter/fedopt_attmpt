import torch.optim as optim
import torch.nn as nn
import torch
from collections import Counter
import logging
from copy import deepcopy
import os

class client(object):
    def __init__(self,
                 rank,
                 world_size,
                 model,
                 args):

        self.rank = rank
        self.world_size = world_size

        self.model = model
        self.optimizer = optim.SGD(model.parameters(),
                                   lr=args.local_lr)

        self.criterion = nn.CrossEntropyLoss()

        self.epochs = args.epochs

        self.start_logger()
        self.load_local_data(args)


    def get_global_weights(self,global_weights):
        self.model.load_state_dict(global_weights)
        self.logger.info("Loaded Global Model Weights")

    def send_pseudo_grad(self):
        self.logger.info("Send Client Local Pseudo-Gradient")
        return self.pseudo_grad

    def send_num_train(self):
        self.logger.info("Send Client Number of Training Instances")
        return self.n_train

    def train(self):
        old_weights = deepcopy(self.model.state_dict())
        for epoch in range(self.epochs):  # loop over the dataset multiple times

            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        self.logger.info('Finished Training')
        new_weights = self.model.state_dict()

        with torch.no_grad():
            self.pseudo_grad = {param_name: (old_weights[param_name].data - new_weights[param_name]) for param_name in new_weights.keys()}

        self.logger.info('Calculated Pseudo Gradient')
    def evaluate(self):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.logger.info(f"Client {self.rank} Evaluating Data: {round(correct / total, 3)}")
        return correct, total

    def start_logger(self):
        self.logger = logging.getLogger(f"client{self.rank}")
        self.logger.setLevel(logging.INFO)

        format = logging.Formatter("%(asctime)s: %(message)s")

        fh = logging.FileHandler(filename=f"logs/client{self.rank}.log",mode='w')
        fh.setFormatter(format)
        fh.setLevel(logging.INFO)


        self.logger.addHandler(fh)

    def load_local_data(self,args):
        self.trainloader = torch.load(os.path.join(args.datapath ,f"data_worker{self.rank}_train.pt"))
        self.testloader = torch.load(os.path.join(args.datapath ,f"data_worker{self.rank}_test.pt"))

        self.n_train = len(self.trainloader.dataset)
        self.logger.info("Local Data Statistics:")
        self.logger.info("Dataset Size: {:.2f}".format(self.n_train))
        self.logger.info(dict(Counter(self.trainloader.dataset[:][1].numpy().tolist())))