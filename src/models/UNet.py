import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = ConvBlock(3, 64)
        self.conv1_2 = ConvBlock(64, 64)
        self.down1 = DownSampleBlock(2)

        self.conv2_1 = ConvBlock(64, 128)
        self.conv2_2 = ConvBlock(128, 128)
        self.down2 = DownSampleBlock(2)

        self.conv3_1 = ConvBlock(128, 256)
        self.conv3_2 = ConvBlock(256, 256)
        self.down3 = DownSampleBlock(2)

        self.conv4_1 = ConvBlock(256, 512)
        self.conv4_2 = ConvBlock(512, 512)
        self.down4 = DownSampleBlock(2)

        self.conv5_1 = ConvBlock(512, 1024)
        self.conv5_2 = ConvBlock(1024, 512)

        self.up1 = UpSampleBlock(512)
        self.conv6_1 = ConvBlock(2*512, 512)
        self.conv6_2 = ConvBlock(512, 256)

        self.up2 = UpSampleBlock(256)
        self.conv7_1 = ConvBlock(2*256, 256)
        self.conv7_2 = ConvBlock(256, 128)

        self.up3 = UpSampleBlock(128)
        self.conv8_1 = ConvBlock(2*128, 128)
        self.conv8_2 = ConvBlock(128, 64)

        self.up4 = UpSampleBlock(64)
        self.conv9_1 = ConvBlock(2*64, 64)
        self.conv9_2 = ConvBlock(64, 1)


    def forward(self, x):
        x = self.conv1_1(x)
        x_cache1 = self.conv1_2(x)
        x = self.down1(x_cache1)

        x = self.conv2_1(x)
        x_cache2 = self.conv2_2(x)
        x = self.down2(x_cache2)

        x = self.conv3_1(x)
        x_cache3 = self.conv3_2(x)
        x = self.down3(x_cache3)

        x = self.conv4_1(x)
        x_cache4 = self.conv4_2(x)
        x = self.down4(x_cache4)

        x = self.conv5_1(x)
        x = self.conv5_2(x)

        x = self.up1(x)
        x = torch.cat((x_cache4, x), dim=1)
        x = self.conv6_1(x)
        x = self.conv6_2(x)

        x = self.up2(x)
        x = torch.cat((x_cache3, x), dim=1)
        x = self.conv7_1(x)
        x = self.conv7_2(x)

        x = self.up3(x)
        x = torch.cat((x_cache2, x), dim=1)
        x = self.conv8_1(x)
        x = self.conv8_2(x)

        x = self.up4(x)
        x = torch.cat((x_cache1, x), dim=1)
        x = self.conv9_1(x)
        x = self.conv9_2(x)

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
