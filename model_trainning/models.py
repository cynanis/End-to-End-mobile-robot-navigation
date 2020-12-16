import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.1), use_shortcut=False, norm_layer=None, dropout=None):
        super(BasicBlock, self).__init__()

        self.use_dropout = True
        if dropout is None:
            self.use_dropout = False
            norm_layer = nn.BatchNorm1d

        self.use_normalization = True
        if norm_layer is None:
            self.use_normalization = False
            norm_layer = nn.BatchNorm1d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(
            in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.bn1 = norm_layer(out_channel)
        self.activation = activation
        self.conv2 = nn.Conv1d(
            out_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.bn2 = norm_layer(out_channel)
        self.dropout = dropout
        self.downsample = None
        self.use_shortcut = use_shortcut
        if self.use_shortcut:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1),
                norm_layer(out_channel))

    def forward(self, x):
        if self.use_shortcut:
            identity = x

        out = self.conv1(x)
        if self.use_normalization:
            out = self.bn1(out)
        out = self.activation(out)
        if self.use_dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        if self.use_normalization:
            out = self.bn2(out)

        if self.use_shortcut:
            identity = self.downsample(x)
            out += identity

        out = self.activation(out)
        if self.use_dropout:
            out = self.dropout(out)

        return out


class Net_CNN(nn.Module):
    def __init__(self):
        super(Net_CNN, self).__init__()

        # activation functions
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.activation2 = nn.ReLU()
        self.activation3 = nn.Tanh()

        # conv block 1
        self.conv1_1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=7, padding=3)  # 360*16
        self.block1 = BasicBlock(in_channel=32, out_channel=32, kernel=7,
                                 stride=1, padding=3, activation=self.activation, dropout=nn.Dropout(0.1))
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 180*32
        # conv block 2

        self.block2_1 = BasicBlock(in_channel=32, out_channel=64, kernel=5, stride=1,
                                   padding=2, activation=self.activation, dropout=nn.Dropout(0.1))  # 180*64
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 90*64

        # conv block 3
        self.block3_1 = BasicBlock(in_channel=64, out_channel=128, kernel=3, stride=1,
                                   padding=1, activation=self.activation, dropout=nn.Dropout(0.1))  # 90*128
        self.max_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 45*128

        # conv block 4
        self.block4_1 = BasicBlock(in_channel=128, out_channel=256, kernel=3, stride=1,
                                   padding=1, activation=self.activation, dropout=nn.Dropout(0.1))  # 45*256
        self.max_pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # 22*256

        # branch2
        self.branch2_fc1 = nn.Linear(in_features=3, out_features=64)
        self.branch2_fc2 = nn.Linear(in_features=64, out_features=256)
        self.branch2_fc3 = nn.Linear(in_features=256, out_features=512)

        # main branch
        self.cat_BN = nn.BatchNorm1d(22*256+512)
        self.Dopoutfc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=22*256+512, out_features=1024)
        self.fc2_BN = nn.BatchNorm1d(1024)
        self.Dopoutfc2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3_BN = nn.BatchNorm1d(1024)
        self.Dopoutfc3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(in_features=1024, out_features=512)
        self.fc4_BN = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(in_features=512, out_features=2)

    def forward(self, scan, goal):
        # block 1
        x = self.conv1_1(scan)
        x = self.activation(x)
        x = self.block1(x)
        x = self.max_pool1(x)

        # block 2
        x = self.block2_1(x)
        x = self.max_pool2(x)

        # block 3
        x = self.block3_1(x)
        x = self.max_pool3(x)

        # block 4
        x = self.block4_1(x)
        x = self.max_pool4(x)
        x = torch.flatten(x, start_dim=1)

        # branch2
        x1 = self.activation2(self.branch2_fc1(goal))
        x1 = self.activation2(self.branch2_fc2(x1))
        x1 = self.activation2(self.branch2_fc3(x1))

        # main branch
        x = torch.cat((x, x1), dim=1)
        x = self.Dopoutfc1(x)
        x = self.activation(self.fc2_BN(self.fc2(x)))
        x = self.Dopoutfc2(x)
        x = self.activation(self.fc3_BN(self.fc3(x)))
        x = self.Dopoutfc3(x)
        x = self.activation(self.fc4_BN(self.fc4(x)))
        x = self.fc5(x)
        return x


# Temporal Network
class Net_CNN_LSTM(nn.Module):
    def __init__(self, sequence_length=8, mode="train", bidirectional=False):

        super(Net_CNN_LSTM, self).__init__()

        # mode of the model
        self.mode = mode

        # activation funcitons
        self.activation1 = nn.LeakyReLU(negative_slope=0.1)
        self.activation2 = nn.ReLU()
        self.activation3 = nn.Tanh()

        # conv block 1
        self.conv1_1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=7, padding=3)  # 360*16
        self.block1 = BasicBlock(in_channel=32, out_channel=32,
                                 kernel=7, stride=1, padding=3, activation=self.activation1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 180*32

        # conv block 2
        self.block2_1 = BasicBlock(in_channel=32, out_channel=64, kernel=5,
                                   stride=1, padding=2, activation=self.activation1)  # 180*64
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 90*64

        # conv block 3
        self.block3_1 = BasicBlock(in_channel=64, out_channel=128, kernel=3,
                                   stride=1, padding=1, activation=self.activation1)  # 90*128
        self.max_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 45*128

        # conv block 4
        self.block4_1 = BasicBlock(in_channel=128, out_channel=256, kernel=3,
                                   stride=1, padding=1, activation=self.activation1)  # 45*256
        self.max_pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # 22*256

        # fc
        self.fc1 = nn.Linear(in_features=22*256, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=256)

        # goals fc layers
        self.fc1goal = nn.Linear(3, 64)
        self.fc2goal = nn.Linear(64, 128)

        self.seq_length = sequence_length
        if bidirectional:
            h_size = 256
            in_size = h_size*2
        else:
            h_size = 256
            in_size = h_size
        
       
        self.lstm0 = nn.LSTM(input_size=128+256,
                             hidden_size=h_size, batch_first=True, num_layers=2, bidirectional=bidirectional)
        self.lstm1 = nn.LSTM(input_size=128+256+in_size,
                             hidden_size=h_size, batch_first=True, num_layers=2, bidirectional=bidirectional)

        self.fcl1 = nn.Linear(in_features=in_size, out_features=256)
        self.fcl2 = nn.Linear(in_features=256, out_features=128)
        self.fcl3 = nn.Linear(in_features=128, out_features=2)

    def forward(self, scan, goal, prev_features=None):
        """
          hidden is a tuple  ( h shape (1, 16, 2),c shape (1, 16, 2))
          """
        # block 1
        x = self.conv1_1(scan)
        x = self.activation3(x)
        x = self.block1(x)
        x = self.max_pool1(x)

        # block 2
        x = self.block2_1(x)
        x = self.max_pool2(x)

        # block 3
        x = self.block3_1(x)
        x = self.max_pool3(x)

        # block 4
        x = self.block4_1(x)
        x = self.max_pool4(x)

        # fc
        x = torch.flatten(x, start_dim=1)
        x = self.activation3(self.fc1(x))
        x = self.activation3(self.fc2(x))
        x = self.activation3(self.fc3(x))

        # goals up_sampler
        goal = self.activation3(self.fc1goal(goal))
        goal = self.activation3(self.fc2goal(goal))
        if self.mode == "train":
            # reshape data
            mini_batch = int(goal.size()[0]/self.seq_length)
            # shape (mini_batch,sequence,input)
            x = x.view((mini_batch, self.seq_length, -1))
            # shape (mini_batch,sequence,input)
            x1 = goal.view((mini_batch, self.seq_length, -1))
            x = torch.cat((x, x1), dim=2)
            # x = torch.transpose(x, 1, 0).contiguous() #shape (sequence, mini_batch, input)

            # LSTM
            for i in range(self.seq_length):
                if i == 0:
                    in_ = torch.unsqueeze(x[:, i, :], dim=1)
                    out, h = self.lstm0(in_)
                else:
                    in_ = torch.cat(
                        (torch.unsqueeze(x[:, i, :], dim=1), out), dim=2)
                    out, h = self.lstm1(in_, h)  # (mini_batch,1,input)

            #x = torch.reshape(out, (mini_batch*self.seq_length, -1))
            x = torch.squeeze(out)  # (mini_batch,input)

            x = self.activation3(self.fcl1(x))
            x = self.activation3(self.fcl2(x))
            x = self.fcl3(x)
            return x
        else:
            x = torch.cat((x, goal), dim=1)  # shape (1,input)
            x = torch.unsqueeze(x, dim=1)  # shape (1,1,259)
            # add previous features
            if prev_features == None:
                features = x.detach()
                return None, features

            elif len(prev_features[0, :, 0]) < self.seq_length:
                features = torch.cat((prev_features, x), dim=1).detach()
                return None, features

            elif len(prev_features[0, :, 0]) == self.seq_length:
                features = torch.cat(
                    (prev_features[:, 1:, :], x), dim=1).detach()
            else:
                print("Error to many sequences")
                return

            # LSTM
            # out, _ = self.lstm(features)  # (mini_batch,sequence,input)
            for i in range(self.seq_length):
                if i == 0:
                    in_ = torch.unsqueeze(features[:, i, :], dim=1)
                    out, h = self.lstm0(in_)
                else:
                    in_ = torch.cat(
                        (torch.unsqueeze(features[:, i, :], dim=1), out), dim=2)
                    out, h = self.lstm1(in_, h)  # (mini_batch,1,input)
            ## out = torch.reshape(out, (self.seq_length, -1))
            # out = out[:, -1, :]  # select the last steering element
            out = torch.squeeze(out)  # (mini_batch,input)
            x = self.activation3(self.fcl1(out))
            x = self.activation3(self.fcl2(x))
            x = self.fcl3(x)
            return x, features


# Temporal with mish

class Net_CNN_LSTM2(nn.Module):
    def __init__(self, sequence_length=8, mode="train", bidirectional=False):

        super(Net_CNN_LSTM2, self).__init__()

        # mode of the model
        self.mode = mode

        # activation funcitons
        self.activation1 = nn.LeakyReLU(negative_slope=0.1)
        self.activation2 = nn.ReLU()
        self.activation3 = nn.Tanh()
        self.activation4 = self.mish

        # conv block 1
        self.conv1_1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=7, padding=3)  # 360*16
        self.block1 = BasicBlock(in_channel=32, out_channel=32,
                                 kernel=7, stride=1, padding=3, activation=self.activation4)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 180*32

        # conv block 2
        self.block2_1 = BasicBlock(in_channel=32, out_channel=64, kernel=5,
                                   stride=1, padding=2, activation=self.activation4)  # 180*64
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 90*64

        # conv block 3
        self.block3_1 = BasicBlock(in_channel=64, out_channel=128, kernel=3,
                                   stride=1, padding=1, activation=self.activation4)  # 90*128
        self.max_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 45*128

        # conv block 4
        self.block4_1 = BasicBlock(in_channel=128, out_channel=256, kernel=3,
                                   stride=1, padding=1, activation=self.activation4)  # 45*256
        self.max_pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # 22*256

        # fc
        self.fc1 = nn.Linear(in_features=22*256, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=256)

        # goals fc layers
        self.fc1goal = nn.Linear(3, 64)
        self.fc2goal = nn.Linear(64, 128)

        self.seq_length = sequence_length
        if bidirectional:
            h_size = 256
            in_size = h_size*2
        else:
            h_size = 256
            in_size = h_size
        
       
        self.lstm0 = nn.LSTM(input_size=128+256,
                             hidden_size=h_size, batch_first=True, num_layers=2, bidirectional=bidirectional)
        self.lstm1 = nn.LSTM(input_size=128+256+in_size,
                             hidden_size=h_size, batch_first=True, num_layers=2, bidirectional=bidirectional)

        self.fcl1 = nn.Linear(in_features=in_size, out_features=256)
        self.fcl2 = nn.Linear(in_features=256, out_features=128)
        self.fcl3 = nn.Linear(in_features=128, out_features=2)

    def forward(self, scan, goal, prev_features=None):
        """
          hidden is a tuple  ( h shape (1, 16, 2),c shape (1, 16, 2))
          """
        # block 1
        x = self.conv1_1(scan)
        x = self.activation4(x)
        x = self.block1(x)
        x = self.max_pool1(x)

        # block 2
        x = self.block2_1(x)
        x = self.max_pool2(x)

        # block 3
        x = self.block3_1(x)
        x = self.max_pool3(x)

        # block 4
        x = self.block4_1(x)
        x = self.max_pool4(x)

        # fc
        x = torch.flatten(x, start_dim=1)
        x = self.activation4(self.fc1(x))
        x = self.activation4(self.fc2(x))
        x = self.activation4(self.fc3(x))

        # goals up_sampler
        goal = self.activation4(self.fc1goal(goal))
        goal = self.activation4(self.fc2goal(goal))
        if self.mode == "train":
            # reshape data
            mini_batch = int(goal.size()[0]/self.seq_length)
            # shape (mini_batch,sequence,input)
            x = x.view((mini_batch, self.seq_length, -1))
            # shape (mini_batch,sequence,input)
            x1 = goal.view((mini_batch, self.seq_length, -1))
            x = torch.cat((x, x1), dim=2)
            # x = torch.transpose(x, 1, 0).contiguous() #shape (sequence, mini_batch, input)

            # LSTM
            for i in range(self.seq_length):
                if i == 0:
                    in_ = torch.unsqueeze(x[:, i, :], dim=1)
                    out, h = self.lstm0(in_)
                else:
                    in_ = torch.cat(
                        (torch.unsqueeze(x[:, i, :], dim=1), out), dim=2)
                    out, h = self.lstm1(in_, h)  # (mini_batch,1,input)

            #x = torch.reshape(out, (mini_batch*self.seq_length, -1))
            x = torch.squeeze(out)  # (mini_batch,input)

            x = self.activation3(self.fcl1(x))
            x = self.activation3(self.fcl2(x))
            x = self.fcl3(x)
            return x
        else:
            x = torch.cat((x, goal), dim=1)  # shape (1,input)
            x = torch.unsqueeze(x, dim=1)  # shape (1,1,259)
            # add previous features
            if prev_features == None:
                features = x.detach()
                return None, features

            elif len(prev_features[0, :, 0]) < self.seq_length:
                features = torch.cat((prev_features, x), dim=1).detach()
                return None, features

            elif len(prev_features[0, :, 0]) == self.seq_length:
                features = torch.cat(
                    (prev_features[:, 1:, :], x), dim=1).detach()
            else:
                print("Error to many sequences")
                return

            # LSTM
            # out, _ = self.lstm(features)  # (mini_batch,sequence,input)
            for i in range(self.seq_length):
                if i == 0:
                    in_ = torch.unsqueeze(features[:, i, :], dim=1)
                    out, h = self.lstm0(in_)
                else:
                    in_ = torch.cat(
                        (torch.unsqueeze(features[:, i, :], dim=1), out), dim=2)
                    out, h = self.lstm1(in_, h)  # (mini_batch,1,input)
            ## out = torch.reshape(out, (self.seq_length, -1))
            # out = out[:, -1, :]  # select the last steering element
            out = torch.squeeze(out)  # (mini_batch,input)
            x = self.activation3(self.fcl1(out))
            x = self.activation3(self.fcl2(x))
            x = self.fcl3(x)
            return x, features
    def mish(self,x):
      return  (x*torch.tanh(F.softplus(x)))     