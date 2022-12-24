# # Model U Net

# import torch
# import torch.nn as nn


# class UNet(nn.Module):
#     def __init__(self, init_features ,in_channels=3, out_channels=1):
#         super(UNet, self).__init__()

#         features = init_features
        
#         self.encoder1 = UNet.conv_block(in_channels , features)
#         self.encoder2 = UNet.conv_block(features,features*2)
#         self.encoder3 = UNet.conv_block(features*2, features*4)
#         self.encoder4 = UNet.conv_block(features*4,features*8)
#         self.bottleneck = UNet.conv_block(features*8,features*16)
#         self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
#         self.decoder4 = UNet.conv_block(features*16,features*8)
#         self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
#         self.decoder3 = UNet.conv_block(features*8,features*4)
#         self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
#         self.decoder2 = UNet.conv_block(features*4,features*2)
#         self.upconv1 = nn.ConvTranspose2d(features * 2, features , kernel_size=2, stride=2)
#         self.decoder1 = UNet.conv_block(features*2,features)
#         self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

#     def forward(self,img_map):
        
#         enc1 = self.encoder1(img_map)
#         print(enc1.size,'enc1')
#         enc2 = self.encoder2(enc1)
#         print(enc2.size,'enc2')
#         enc3 = self.encoder3(enc2)
#         print(enc3.size,'enc3')
#         enc4 = self.encoder4(enc3)
#         print(enc4.size,'enc4')

#         bottleneck = self.bottleneck(enc4)

#         dec4 = self.upconv4(bottleneck)
       
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         print(dec4.size,'dec4')
#         dec3 = self.upconv3(dec4)
        
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
       
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         print(dec2.size,'dec2')
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
#         print(dec1.size,'dec1')

#         return torch.sigmoid(self.conv(dec1))


#     @staticmethod
#     def conv_block(in_channels,features):
#         return nn.Sequential( 
#             nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(features),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, stride=1, bias=False),
#             nn.BatchNorm2d(features),
#             nn.ReLU(inplace = True),
#             nn.MaxPool2d(kernel_size=2)
#             )
        

        
# x = torch.rand(1, 3, 224, 224)

# u = UNet(init_features=32)
# y = u.forward(x)


# Model U Net

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, init_features ,in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        features = init_features
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.encoder1 = UNet.conv_block(in_channels , features)
        self.encoder2 = UNet.conv_block(features,features*2)
        self.encoder3 = UNet.conv_block(features*2, features*4)
        self.encoder4 = UNet.conv_block(features*4,features*8)
        self.bottleneck = UNet.conv_block(features*8,features*16)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet.conv_block(features*16,features*8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet.conv_block(features*8,features*4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet.conv_block(features*4,features*2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features , kernel_size=2, stride=2)
        self.decoder1 = UNet.conv_block(features*2,features)
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self,img_map):

        
        
        enc1 = self.encoder1(img_map)
        # print(enc1.shape,'enc1')
        
        enc2 = self.encoder2(self.pool(enc1))
        # print(enc2.shape,'enc2')
        
      
        enc3 = self.encoder3(self.pool(enc2))
        # print(enc3.shape,'enc3')
        
        # enc3 = self.pool(enc3)
        # print(enc3.shape,'enc3')
        enc4 = self.encoder4(self.pool(enc3))
        # print(enc4.shape,'enc4')
      
    
        # print(enc4.shape,'enc4')

        

        bottleneck = self.bottleneck(self.pool(enc4))
        # print(bottleneck.shape,'bott')

        dec4 = self.upconv4(bottleneck)
        # print(dec4.shape,'dec4upcon')
        
        # y = UNet.trans(s,dec4.shape[2])
        # print(y.shape,'crop')

        enc4 = UNet.trans(enc4,dec4.shape[2])
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)
        # print(dec4.shape,'dec4')
        dec3 = self.upconv3(dec4)
        
        enc3 = UNet.trans(enc3,dec3.shape[2])
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)


       
        dec2 = self.upconv2(dec3)
        enc2 = UNet.trans(enc2,dec2.shape[2])
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        # print(dec2.shape,'dec2')

        dec1 = self.upconv1(dec2)
        enc1 = UNet.trans(enc1,dec1.shape[2])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        # print(dec1.shape,'dec1')
        # print(self.conv(dec1).shape)
        return torch.sigmoid(self.conv(dec1))


    @staticmethod
    def conv_block(in_channels,features):
        return nn.Sequential( 
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace = True),
            )

    @staticmethod    
    def trans(img,shape):
      x = img.shape[2]
  
      u = (x-shape)/2
     
      a = int(u-1)
      b = int(a+shape)
      return img[:,:,a:b,a:b]
        

