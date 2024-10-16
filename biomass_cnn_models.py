import torch.nn as nn
import torch

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