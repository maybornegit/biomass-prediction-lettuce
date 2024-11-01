import torch.nn as nn
import torch
from torchvision import models, transforms
import numpy as np

class MAPE(nn.Module):
    def __init__(self):
        super(MAPE, self).__init__()

    def forward(self, y_true, y_pred):
        epsilon = 1e-6
        return torch.mean(torch.abs((y_true-y_pred)/(y_true+epsilon)))*100
class CNN_128(nn.Module):
    def __init__(self):
        super(CNN_128, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=5, stride=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(512 * 1 * 1, 1)  # Adjusted for regression output size

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = x.view(-1, 512 * 1 * 1)  # Flatten the tensor
        x = self.dropout(x)
        x = self.fc(x)
        return torch.squeeze(x)


class CNN_640(nn.Module):
    def __init__(self):
        super(CNN_640, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=5, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(128, 256, kernel_size=5, stride=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU()
        self.pool7 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(512 * 1 * 1, 1)  # Adjusted for regression output size

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.pool6(self.relu6(self.bn6(self.conv6(x))))
        x = self.pool7(self.relu7(self.bn7(self.conv7(x))))
        x = x.view(-1, 512 * 1 * 1)  # Flatten the tensor
        x = self.dropout(x)
        x = self.fc(x)
        return torch.squeeze(x)


class CNN_Delta(nn.Module):
    def __init__(self):
        super(CNN_Delta, self).__init__()
        self.conv1 = nn.Conv2d(8, 8, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=5, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(128, 256, kernel_size=5, stride=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU()
        self.pool7 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(512 * 1 * 1, 1)  # Adjusted for regression output size

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.pool6(self.relu6(self.bn6(self.conv6(x))))
        x = self.pool7(self.relu7(self.bn7(self.conv7(x))))
        x = x.view(-1, 512 * 1 * 1)  # Flatten the tensor
        x = self.dropout(x)
        x = self.fc(x)
        return torch.squeeze(x)

class ResNet640(nn.Module):
    def __init__(self):
        super(ResNet640, self).__init__()
        self.resnetD = models.resnet50(weights='DEFAULT')
        self.resnetRGB = models.resnet50(weights='DEFAULT')

        # Modify the first convolution layer to accept 1 input channel
        self.resnetRGB.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnetD.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the fully connected layer with two separate output layers
        num_featuresRGB = self.resnetRGB.fc.in_features
        self.resnetRGB.fc = nn.Identity()  # Remove the fully connected layer
        num_featuresD = self.resnetD.fc.in_features
        self.resnetD.fc = nn.Identity()  # Remove the fully connected layer

        # Create two linear layers for the two output vectors
        self.fc1 = nn.Linear(num_featuresRGB, 1000)
        self.fc2 = nn.Linear(num_featuresD, 1000)

        self.fc3 = nn.Linear(2000,1000)
        self.fc4 = nn.Linear(1000,1)

        if torch.cuda.is_available():
            self.resnetRGB = self.resnetRGB.to('cuda')
            self.resnetD = self.resnetD.to('cuda')
            self.fc3 = self.fc3.to('cuda')
            self.fc4 = self.fc4.to('cuda')
            self.device = torch.device("cuda:0")

    def forward(self,x):
        preprocessRGB = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 640x640
            transforms.Normalize(  # Normalize the image
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        preprocessD = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 640x640
            transforms.Normalize(  # Normalize the image
                mean=[0.5],
                std=[0.5]
            )
        ])
        x_RGB = x[:,:3,:,:]
        x_RGB = preprocessRGB(x_RGB).to(self.device)
        x_D = x[:,3,:,:].unsqueeze(1)
        x_D = preprocessD(x_D).to(self.device)
        # print(x_RGB.shape)
        # print(x_D.shape)
        features_RGB = self.resnetRGB(x_RGB)
        features_D = self.resnetD(x_D)
        # print(features_RGB.shape)
        # print(features_D.shape)
        new_featuresRGB = self.fc1(features_RGB)
        new_featuresD = self.fc2(features_D)
        # print(features_RGB.shape)
        # print(features_D.shape)

        combined_features = torch.cat((new_featuresRGB, new_featuresD), dim=1)
        # print(combined_features.shape)
        layer_3 = self.fc3(combined_features)
        output = self.fc4(layer_3)
        # print(layer_3.shape)
        # print(output.shape)
        # print(output)
        return output


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Define the layers
        self.layer1 = nn.Linear(2, 50)  # Input layer
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 50)
        self.layer4 = nn.Linear(50, 50)
        self.layer5 = nn.Linear(50, 50)  # Last hidden layer
        self.output_layer = nn.Linear(50, 1)  # Output layer

    def forward(self, x):
        # Define the forward pass
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = torch.sigmoid(self.output_layer(x))  # Sigmoid for binary output
        return x
