import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)

        self.bottleneck = self.conv_block(256, 512)

        self.upconv3 = self.upconv_block(512, 256)
        self.follow_conv3 = self.conv_block(512, 256)

        self.upconv2 = self.upconv_block(256, 128)
        self.follow_conv2 = self.conv_block(256, 128)

        self.upconv1 = self.upconv_block(128, 64)
        self.follow_conv1 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        c1 = self.encoder1(x)
        p1 = nn.MaxPool2d(kernel_size=2, stride=2)(c1)

        c2 = self.encoder2(p1)
        p2 = nn.MaxPool2d(kernel_size=2, stride=2)(c2)

        c3 = self.encoder3(p2)
        p3 = nn.MaxPool2d(kernel_size=2, stride=2)(c3)

        bn = self.bottleneck(p3)

        u3 = self.upconv3(bn)
        u3 = torch.cat([u3, c3], dim=1)
        u3 = self.follow_conv3(u3)

        u2 = self.upconv2(u3)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.follow_conv2(u2)

        u1 = self.upconv1(u2)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.follow_conv1(u1)

        output = self.final_conv(u1)
        return output + x

# model = UNet().to(device)
# print(model)