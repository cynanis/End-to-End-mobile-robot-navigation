from __future__ import print_function
import numpy as np
import torch
import torch.optim as optim
import time
from loss_functions import MSELoss, RMSELoss


class TrainClass():
    def __init__(self, Net, optimizer, trainloader, validloader, criterion=MSELoss,
                 model_path='./deepplanner.pth', checkpoint_=None, batch_size=128, sequence_length=8):
        """
        args:
        Net= the model to be trained
        optimizer 
        trainloader = the instance of the trainning data loader
        validloader = the instance of the testing data loader
        criterion = the LOSS fucntion CLass like MSELoss or MAELoss present in lossfunctions.py
        model_path = path to our model
        checkpoint = checkpoint instance of previously trained model
        batch_size = the size of the batch of data in data loader
        sequence_length = the number of lstm units in the case the LSTM network


        """

        self.model = Net
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainloader = trainloader
        self.validloader = validloader
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.checkpoint = checkpoint_
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainData()

    def train(self, epochs, single_optimize=False):
        self.preInformation()
        # model to device
        self.model.to(self.device)
        if self.checkpoint != None:
            self.loadCheckpointData()
        # init loss function
        loss_func = self.criterion()

        for epoch in range(self.start_epochs, self.start_epochs+epochs):
            # decay learning rate
            # if epoch%20==19 and optimizer.param_groups[0]["lr"] > 0.0002:
            #   optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] - 0.0002

            loss_train = 0.0
            loss_linear_train = 0.0
            loss_angular_train = 0.0
            loss_valid = 0.0
            loss_linear_valid = 0.0
            loss_angular_valid = 0.0

            start_loop = time.time()

            print('Epoch: %d/%d' % (epoch + 1, self.start_epochs+epochs))

            # train
            self.model.train()
            for i, samples in enumerate(self.trainloader):

                scan, goal, steering = samples["scan"].to(self.device, dtype=torch.float), samples["goal"].to(
                    self.device, dtype=torch.float), samples["steering"].to(self.device, dtype=torch.float)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                if single_optimize:
                    # shape (batch,sequence,input)
                    mini_batch = int(goal.size()[0]/self.sequence_length)
                    steering = steering.view(
                        (mini_batch, self.sequence_length, -1))
                    steering = steering[:, -1, :]

                # forward + backward + optimize
                output = self.model(scan, goal)
                loss, linear_loss, angular_loss = loss_func(output, steering)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 5)

                self.optimizer.step()
                loss_train += loss.item()
                loss_linear_train += linear_loss.item()
                loss_angular_train += angular_loss.item()

                # if i % 2000:
                #     # print losses
                #     print(' [Train_loss: %.5f, angular_loss: %.5f , angular_loss: %.5f ' % (
                #         loss.item(), linear_loss.item(), angular_loss.item()))

            # test
            self.model.eval()
            with torch.no_grad():
                for samples in self.validloader:

                    scan, goal, steering = samples["scan"].to(self.device, dtype=torch.float), samples["goal"].to(
                        self.device, dtype=torch.float), samples["steering"].to(self.device, dtype=torch.float)

                    if single_optimize:
                        # shape (batch,sequence,input)
                        mini_batch = int(goal.size()[0]/self.sequence_length)
                        steering = steering.view(
                            (mini_batch, self.sequence_length, -1))
                        steering = steering[:, -1, :]

                    output = self.model(scan, goal)

                    loss, linear_loss, angular_loss = loss_func(
                        output, steering)

                    loss_valid += loss.item()
                    loss_linear_valid += linear_loss.item()
                    loss_angular_valid += angular_loss.item()

            end_loop = time.time()

            # calculate valid and train losses
            data_train_len = (len(self.trainloader))
            data_valid_len = (len(self.validloader))
            loss_train_ = loss_train/data_train_len
            loss_linear_train_ = loss_linear_train/data_train_len
            loss_angular_train_ = loss_angular_train/data_train_len
            loss_valid_ = loss_valid/data_valid_len
            loss_linear_valid_ = loss_linear_valid/data_valid_len
            loss_angular_valid_ = loss_angular_valid/data_valid_len
            # if losses are squared do sqrt(loss)
            if self.criterion == MSELoss:
                loss_train_ = np.sqrt(loss_train_)
                loss_linear_train_ = np.sqrt(loss_linear_train_)
                loss_angular_train_ = np.sqrt(loss_angular_train_)
                loss_valid_ = np.sqrt(loss_valid_)
                loss_linear_valid_ = np.sqrt(loss_linear_valid_)
                loss_angular_valid_ = np.sqrt(loss_angular_valid_)

            # print losses
            print('epoch:%d [Train_loss: %.5f, Valid_loss: %.5f ,time:%.5f min, lr: %.5f]' % (epoch+1, loss_train_, loss_valid_, (end_loop -
                                                                                                                                  start_loop)/60, self.optimizer.param_groups[0]["lr"]), "[linear_v: %.5f " % loss_linear_valid_, "angular_v:%.5f" % loss_angular_valid_, "]")

            # append losses
            self.train_losses.append([loss_train_, epoch+1])
            self.train_linear_losses.append([loss_linear_train_, epoch+1])
            self.train_angular_losses.append([loss_angular_train_, epoch+1])
            self.valid_losses.append([loss_valid_, epoch+1])
            self.valid_linear_losses.append([loss_linear_valid_, epoch+1])
            self.valid_angular_losses.append([loss_angular_valid_, epoch+1])
            self.lr_rates.append(
                [self.optimizer.param_groups[0]["lr"], epoch+1])

            # save model
            if loss_valid_ < self.prev_valid_loss:
                self.prev_valid_loss = loss_valid_
                self.saveModel(epoch=epoch, onTest=True)

            if loss_train_ < self.prev_train_loss:
                self.prev_train_loss = loss_train_
                self.saveModel(epoch=epoch, onTest=False)

        print('finished Training at: ', time.asctime(
            time.localtime(time.time())))

    def preInformation(self):
        # pre training information
        if self.checkpoint == None:
            print("\n Training started from Zero")
        else:
            print("\n Training started from a checkpoint")

        localtime = time.asctime(time.localtime(time.time()))
        print("\n Training started at ", localtime)
        print("Device ", self.device)

    def loadCheckpointData(self):

        if self.checkpoint is not None:
            self.train_losses = self.checkpoint["train_losses"]
            self.train_linear_losses = self.checkpoint["train_linear"]
            self.train_angular_losses = self.checkpoint["train_angular"]
            self.valid_losses = self.checkpoint["valid_losses"]
            self.valid_linear_losses = self.checkpoint["valid_linear"]
            self.valid_angular_losses = self.checkpoint["valid_angular"]
            self.start_epochs = self.checkpoint["epoch"]
            self.lr_rates = self.checkpoint["lr"]
            self.prev_train_loss = self.checkpoint["prev_train_loss"]
            self.prev_valid_loss = self.checkpoint["prev_valid_loss"]
            for i in range(0, self.start_epochs):
                print("epoch: ", i+1, "Train_loss: %.5f" % self.train_losses[i][0], "Valid_loss: %.5f" % self.valid_losses[i][0], "lr: %.5f" % self.lr_rates[i]
                      [0], "[valid_linear: %.5f, valid_angular:%.5f]" % (self.valid_linear_losses[i][0], self.valid_angular_losses[i][0]))

    def trainData(self):
        # load previous training data
        self.train_losses = []
        self.train_linear_losses = []
        self.train_angular_losses = []
        self.valid_losses = []
        self.valid_linear_losses = []
        self.valid_angular_losses = []
        self.lr_rates = []
        self.list_of_losses_valid = []
        self.list_of_losses_train = []
        self.prev_valid_loss = np.inf
        self.prev_train_loss = np.inf
        self.start_epochs = 0

    def saveModel(self, epoch, onTest=True):

        checkpoint = {'input_size': [(1, 360), (1, 3)],
                      'output_size': 2,
                      'device': self.device,
                      'batch_size': self.batch_size,
                      'optimizer': self.optimizer.state_dict(),
                      'epoch': epoch+1,
                      'lr': self.lr_rates,
                      'lr': self.lr_rates,
                      'list_of_losses_valid':  self.list_of_losses_valid,
                      'list_of_losses_train':  self.list_of_losses_train,
                      'train_losses': self.train_losses,
                      "train_linear": self.train_linear_losses,
                      "train_angular": self.train_angular_losses,
                      'valid_losses': self.valid_losses,
                      "valid_linear": self.valid_linear_losses,
                      "valid_angular": self.valid_angular_losses,
                      'prev_valid_loss': self.prev_valid_loss,
                      'prev_train_loss': self.prev_train_loss,
                      'state_dict': self.model.state_dict()}
        if onTest == True:
            torch.save(checkpoint, self.model_path + "_test.pth")
            print("saving model at epoch number %d ....................." % (epoch+1))
        if onTest == False:
            torch.save(checkpoint, self.model_path + "_train.pth")
