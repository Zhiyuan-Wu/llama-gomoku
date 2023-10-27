import sys
sys.path.append('..')

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class GobangNNetArgs:
    game_size: int = 15
    numMCTSSims: int = 300
    cpuct: int = 1
    dirichlet_alpha: float = 0.03
    dirichlet_weight: float = 0.3
    tempereature: float = 1.0
    endGameRewardWeight: float = 1.0

    dropout: float = 0.3
    num_channels: int = 128
    block_num: int = 6

    no_head: bool = False

class GobangNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        self.block_num = args.block_num
        self.num_channels = args.num_channels

        super(GobangNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 5, stride=1, padding=2)

        self.conv_layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])
        for _ in range(args.block_num):
            self.conv_layers.append(nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1))
            self.conv_layers.append(nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1))
            self.bn_layers.append(nn.BatchNorm2d(args.num_channels))
            self.bn_layers.append(nn.BatchNorm2d(args.num_channels))

        if not self.args.no_head:
            self.fc1 = nn.Linear(args.num_channels*(self.board_x)*(self.board_y), 1024)
            self.fc_bn1 = nn.BatchNorm1d(1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc_bn2 = nn.BatchNorm1d(512)

            self.fc3 = nn.Linear(512, self.action_size)
            self.fc4 = nn.Linear(512, 3)

    def forward(self, s):
        s = s.view(-1, 1, self.board_x, self.board_y)
        s = self.conv1(s)

        for i in range(self.block_num):
            _s = F.relu(self.bn_layers[2*i](s))
            _s = self.conv_layers[2*i](_s)
            _s = F.relu(self.bn_layers[2*i+1](_s))
            _s = self.conv_layers[2*i+1](_s)
            s = _s + s

        if not self.args.no_head:
            s = s.view(-1, self.num_channels*(self.board_x)*(self.board_y))

            s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
            s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

            pi = self.fc3(s)                                                                         # batch_size x action_size
            v = self.fc4(s)                                                                          # batch_size x 1

            return F.log_softmax(pi, dim=1), v
        else:
            return s

def loss_pi(targets, outputs):
    ''' This function compute policy loss.
    '''
    return -torch.sum(targets * outputs) / targets.size()[0]

def loss_v(targets, outputs):
    ''' This function compute reward loss.
    '''
    # return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
    return F.cross_entropy(outputs, targets)

def entropy(pi):
    ''' This function compute entropy of a given policy.
    '''
    return - torch.sum(pi * torch.exp(pi)) / pi.size()[0]