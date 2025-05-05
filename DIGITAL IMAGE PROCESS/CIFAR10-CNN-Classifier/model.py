import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        """
        define the layers in your CNN
        """
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 32→16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 16→8

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 8→4

            nn.Conv2d(256, 512, 3, padding=1), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 4→2
        )

        self.gap = nn.AdaptiveAvgPool2d(1)  # 2x2 → 1x1

        self.classifier = nn.Sequential(
            nn.Flatten(),                 # → [batch, 512]
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, 10)           # CIFAR-10
        )

    def init_weights(self):
        """
        optionally initialize weights
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Pass the input images through your CNN to produce the class probabilities

        :param x: batch_size x 3 x 28 x 28 tensor containing the input images
        :return: batch_size x 11 tensor containing the class probabilities for each image
        """
        x = self.conv_block(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x 

class Params:
    def __init__(self):
        self.use_gpu = 1
        self.train = Params.Training()
        self.val = Params.Validation()
        self.test = Params.Testing()
        self.ckpt = Params.Checkpoint()
        self.optim = Params.Optimization()

    def process(self):
        self.val.vis = self.val.vis.ljust(3, '0')
        self.test.vis = self.test.vis.ljust(2, '0')
    
    class Optimization:
        """
        You can modify this class to fine-tune your model
        """
        def __init__(self):
            self.type = 'adam' # You can choose either 'adam' or 'sgd' to play around with optimizer type
            self.lr = 1e-3  # You can play around with the learning rate here
            self.momentum = 0.9
            self.eps = 1e-8
            self.weight_decay = 1e-4  # You can modify weight decay here

    class Training:
        """
        You can modify this class to change parameters for classifier training
        """
        def __init__(self):
            self.probs_data = ''
            self.batch_size = 128  # you can change batch size here
            self.n_workers = 4  
            self.n_epochs = 200  # you can modify number of epoches here

    class Validation:
        def __init__(self):
            self.batch_size = 16
            self.n_workers = 1
            self.gap = 1
            self.ratio = 0.2
            self.vis = '0'
            self.tb_samples = 10

    class Testing:
        def __init__(self):
            self.enable = 0
            self.batch_size = 24
            self.n_workers = 1
            self.vis = '0'

    class Checkpoint:
        """
        :ivar load:
            0: train from scratch;
            1: load checkpoint if it exists and continue training;
            2: load checkpoint and test;

        :ivar save_criteria:  when to save a new checkpoint:
            val-acc: validation accuracy increases;
            val-loss: validation loss decreases;
            train-acc: training accuracy increases;
            train-loss: training loss decreases;

        :ivar path: path to the checkpoint
        """

        def __init__(self):
            self.load = 2
            self.path = './checkpoints/model.pt.114'
            self.save_criteria = [
                'train-acc',
                'train-loss',
                'val-acc',
                'val-loss',
            ]