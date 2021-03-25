
# you can import pretrained models for experimentation & add your own created models
from torchvision.models import resnet18, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import torch
import torch.nn as nn

class fc_model(nn.Module):

    def __init__(self, input_size, num_classes=11, dropout=0.5):
        """
            A linear model for image classification.
        """

        super(fc_model, self).__init__()
        #self.linear = nn.Linear(input_size, num_classes)
        
        #self.dropout = nn.Dropout(dropout)

        # initialize parameters (write code to initialize parameters here)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 56, kernel_size=4, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.BatchNorm2d(56),
            nn.Conv2d(56, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=6, stride=6),

        )
        self.layer2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim = 1)
        )
        

    def forward(self, x):
        """
            feed-forward (vectorized) image into a linear model for classification.   
        """
        x = self.layer1(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        x = self.layer2(x)
        return x

# =======================================
