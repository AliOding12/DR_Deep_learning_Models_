import torch
import torch.nn as nn

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetPlusPlus, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1_0 = conv_block(in_channels, 64)
        self.enc2_0 = conv_block(64, 128)
        self.enc3_0 = conv_block(128, 256)
        self.enc4_0 = conv_block(256, 512)
        
        self.enc1_1 = conv_block(64 + 64, 64)
        self.enc2_1 = conv_block(128 + 128, 128)
        self.enc3_1 = conv_block(256 + 256, 256)
        
        self.enc1_2 = conv_block(64 + 64 + 64, 64)
        self.enc2_2 = conv_block(128 + 128 + 128, 128)
        
        self.enc1_3 = conv_block(64 + 64 + 64 + 64, 64)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1_0 = self.enc1_0(x)
        x2_0 = self.enc2_0(self.pool(x1_0))
        x3_0 = self.enc3_0(self.pool(x2_0))
        x4_0 = self.enc4_0(self.pool(x3_0))
        
        x1_1 = self.enc1_1(torch.cat([x1_0, self.upconv2(x2_0)], dim=1))
        x2_1 = self.enc2_1(torch.cat([x2_0, self.upconv3(x3_0)], dim=1))
        x3_1 = self.enc3_1(torch.cat([x3_0, self.upconv4(x4_0)], dim=1))
        
        x1_2 = self.enc1_2(torch.cat([x1_0, x1_1, self.upconv2(x2_1)], dim=1))
        x2_2 = self.enc2_2(torch.cat([x2_0, x2_1, self.upconv3(x3_1)], dim=1))
        
        x1_3 = self.enc1_3(torch.cat([x1_0, x1_1, x1_2, self.upconv2(x2_2)], dim=1))
        
        return self.final_conv(x1_3)# Add UNet++ model implementation
