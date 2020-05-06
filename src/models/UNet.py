import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(3, 32)
        self.down1 = DownSampleBlock(2)

        self.conv2 = ConvBlock(32, 64)
        self.down2 = DownSampleBlock(2)

        self.conv3 = ConvBlock(64, 128)
        self.down3 = DownSampleBlock(2)

        self.conv4 = ConvBlock(128, 128)

        self.up1 = UpSampleBlock(128)
        self.conv5 = ConvBlock(2*128, 64)

        self.up2 = UpSampleBlock(64)
        self.conv6 = ConvBlock(2*64, 32)

        self.up3 = UpSampleBlock(32)
        self.conv7 = ConvBlock(2*32, 1)


    def forward(self, x):
        x_cache1 = self.conv1(x)
        x = self.down1(x_cache1)

        x_cache2 = self.conv2(x)
        x = self.down2(x_cache2)

        x_cache3 = self.conv3(x)
        x = self.down3(x_cache3)

        x = self.conv4(x)

        x = self.up1(x)
        x = torch.cat((x_cache3, x), dim=1)
        x = self.conv5(x)

        x = self.up2(x)
        x = torch.cat((x_cache2, x), dim=1)
        x = self.conv6(x)

        x = self.up3(x)
        x = torch.cat((x_cache1, x), dim=1)
        x = self.conv7(x)

        return torch.sigmoid(x)


class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel=3):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel, padding=1)
        self.bn = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class DownSampleBlock(nn.Module):
    def __init__(self, kernel=2):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):
        return F.max_pool2d(x, self.kernel)

class UpSampleBlock(nn.Module):
    def __init__(self, num_filters, kernel=3):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(num_filters, num_filters, kernel, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        x = self.tconv(x)
        x = self.bn(x)
        return F.relu(x)
