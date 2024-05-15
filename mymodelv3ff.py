import warnings
import torch.nn.functional as F
from functools import partial
from timm.models.layers import to_2tuple, trunc_normal_
import math
from timm.models.layers import DropPath
from torch.nn import Module
#from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from models import pvt_v2
from transformer import CrossAttentionLayer
from timm.models.vision_transformer import _cfg

class upsample_2x(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, dilation=1):
        super(upsample_2x, self).__init__()
        self.conv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.gn = nn.GroupNorm(32,out_planes)
        self.mish = nn.Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.mish(x)
        x = self.gn(x)
        return x

class downsample_2x(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, dilation=1):
        super(downsample_2x, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.gn = nn.GroupNorm(32,out_planes)
        self.mish = nn.Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.mish(x)
        x = self.gn(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self,c1 = 64,c2 = 128,c3 = 320,c4 = 512):
        super(ChannelAttention, self).__init__()
        self.squeeze = 2

        self.s1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=2, stride=2, padding=0),
            nn.InstanceNorm2d(c1),
            nn.Mish(),
            nn.Conv2d(c1, c1, kernel_size=2, stride=2, padding=0),
            nn.InstanceNorm2d(c1),
            nn.Mish(),
            nn.Conv2d(c1, c1, kernel_size=2, stride=2, padding=0),
            nn.InstanceNorm2d(c1),
            nn.Mish(),
            nn.Conv2d(c1, c1, kernel_size=(11, 1), stride=(11, 1)),
            nn.InstanceNorm2d(c1),
            nn.Mish(),
            nn.Conv2d(c1, c1, kernel_size=(1, 11)),
            nn.Mish()
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=2, stride=2, padding=0),
            nn.InstanceNorm2d(c2),
            nn.Mish(),
            nn.Conv2d(c2, c2, kernel_size=2, stride=2, padding=0),
            nn.InstanceNorm2d(c2),
            nn.Mish(),
            nn.Conv2d(c2, c2, kernel_size=(11, 1), stride=(11, 1)),
            nn.InstanceNorm2d(c2),
            nn.Mish(),
            nn.Conv2d(c2, c2, kernel_size=(1, 11)),
            nn.Mish()
        )
        self.s3 = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=2, stride=2, padding=0),
            nn.InstanceNorm2d(c3),
            nn.Mish(),
            nn.Conv2d(c3, c3, kernel_size=(11, 1), stride=(11, 1)),
            nn.InstanceNorm2d(c3),
            nn.Mish(),
            nn.Conv2d(c3, c3, kernel_size=(1, 11)),
            nn.Mish()
        )
        self.s4 = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=(11, 1), stride=(11, 1)),
            nn.InstanceNorm2d(c4),
            nn.Mish(),
            nn.Conv2d(c4, c4, kernel_size=(1, 11)),
            nn.Mish()
        )
        self.squeeze1 = nn.Sequential(
            nn.Conv2d(c1, c1 // self.squeeze, kernel_size=1),
            nn.GroupNorm(32,c1 // self.squeeze),
            nn.Mish()
        )
        self.squeeze2 = nn.Sequential(
            nn.Conv2d(c2, c2 // self.squeeze, kernel_size=1),
            nn.GroupNorm(32,c2 // self.squeeze),
            nn.Mish()
        )
        self.squeeze3 = nn.Sequential(
            nn.Conv2d(c3, c3 // self.squeeze, kernel_size=1),
            nn.GroupNorm(32,c3 // self.squeeze),
            nn.Mish()
        )
        self.squeeze4 = nn.Sequential(
            nn.Conv2d(c4, c4 // self.squeeze, kernel_size=1),
            nn.GroupNorm(32,c4 // self.squeeze),
            nn.Mish()
        )

    def forward(self,x1,x2,x3,x4):
        y1 = self.s1(x1)
        y2 = self.s2(x2)
        y3 = self.s3(x3)
        y4 = self.s4(x4)
        g1 = torch.softmax(y1,dim=1)
        g2 = torch.softmax(y2,dim=1)
        g3 = torch.softmax(y3, dim=1)
        g4 = torch.softmax(y4, dim=1)
        re1 = x1 * g1.expand_as(x1)
        re2 = x2 * g2.expand_as(x2)
        re3 = x3 * g3.expand_as(x3)
        re4 = x4 * g4.expand_as(x4)
        re1 = self.squeeze1(re1)
        re2 = self.squeeze2(re2)
        re3 = self.squeeze3(re3)
        re4 = self.squeeze4(re4)

        return re1,re2,re3,re4

class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.Mish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=2),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)

class RB1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.Mish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        h = self.in_layers(x)
        return h

class SpatialAttention(nn.Module):
    def __init__(self,c1 = 32,c2 = 64,c3 = 160,c4 = 256):
        super(SpatialAttention, self).__init__()
        self.att21 = CrossAttentionLayer(d_model=c1, nhead=8)
        self.att31 = CrossAttentionLayer(d_model=c1, nhead=8)
        self.att41 = CrossAttentionLayer(d_model=c1, nhead=8)
        self.att32 = CrossAttentionLayer(d_model=c2, nhead=8)
        self.att42 = CrossAttentionLayer(d_model=c2, nhead=8)
        self.att43 = CrossAttentionLayer(d_model=c3, nhead=8)
        self.fu4 = RB1(in_channels=c1+c2+c3+c4, out_channels=c4)
        self.fu3 = RB1(in_channels=c1+c2+c3, out_channels=c3)
        self.fu2 = RB1(in_channels=c1+c2, out_channels=c2)
        # self.ch12 = nn.Conv2d(c2, c1, kernel_size=1)
        # self.ch13 = nn.Conv2d(c3, c1, kernel_size=1)
        # self.ch14 = nn.Conv2d(c4, c1, kernel_size=1)
        self.ch21 = RB1(in_channels=c2, out_channels=c1)
        # self.ch23 = nn.Conv2d(c3, c2, kernel_size=1)
        # self.ch24 = nn.Conv2d(c4, c2, kernel_size=1)
        self.ch31 = RB1(in_channels=c3, out_channels=c1)
        self.ch32 = RB1(in_channels=c3, out_channels=c2)
        #self.ch34 = nn.Conv2d(c4, c3, kernel_size=1)
        self.ch41 = RB1(in_channels=c4, out_channels=c1)
        self.ch42 = RB1(in_channels=c4, out_channels=c2)
        self.ch43 = RB1(in_channels=c4, out_channels=c3)
        # self.patch_embed12 = RB(in_channels=c1, out_channels=c2)
        # self.patch_embed13 = RB(in_channels=c1, out_channels=c3)
        # self.patch_embed14 = RB(in_channels=c1, out_channels=c4)
        # self.patch_embed23 = RB(in_channels=c2, out_channels=c3)
        # self.patch_embed24 = RB(in_channels=c2, out_channels=c4)
        # self.patch_embed34 = RB(in_channels=c3, out_channels=c4)
    def forward(self,x1,x2,x3,x4):
        #y1,y2,y3,y4 = x1,x2,x3,x4
        xc21 = self.ch21(x2)
        atten21 = self.att21(xc21, x1)
        xc31 = self.ch31(x3)
        atten31 = self.att31(xc31, x1)
        xc41 = self.ch41(x4)
        atten41 = self.att41(xc41, x1)
        xc32 = self.ch32(x3)
        atten32 = self.att32(xc32, x2)
        xc42 = self.ch42(x4)
        atten42 = self.att42(xc42,x2)
        xc43 = self.ch43(x4)
        atten43 = self.att43(xc43,x3)
        z4 = self.fu4(torch.cat([atten41, atten42, atten43, x4], dim=1))
        z3 = self.fu3(torch.cat([atten31,atten32,x3], dim=1))
        z2 = self.fu2(torch.cat([atten21,x2], dim=1))

        return x1, z2, z3, z4

class F(nn.Module):
    def __init__(self,c1 = 32,c2 = 64,c3 = 160,c4 = 256):
        super(F, self).__init__()

        self.fuse3 = RB1(in_channels=2*c3, out_channels=c3)
        self.fuse2 = RB1(in_channels=2*c2, out_channels=c2)
        self.fuse1 = RB1(in_channels=2*c1, out_channels=c1)
        self.up4 = upsample_2x(c4, c3)
        self.up3 = upsample_2x(c3, c2)
        self.up2 = upsample_2x(c2, c1)

    def forward(self,x1,x2,x3,x4):
        z4 = self.up4(x4)
        z3 = self.fuse3(torch.cat([x3,z4],dim=1))
        z3 = self.up3(z3)
        z2 = self.fuse2(torch.cat([x2,z3],dim=1))
        z2 = self.up2(z2)
        z1 = self.fuse1(torch.cat([x1,z2],dim=1))

        return z1

class TB(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        checkpoint = torch.load("/home/ge/project/Duibi/models/pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

    def forward(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

class Tnet(nn.Module):
    def __init__(self, class_num=1, **kwargs):
        super(Tnet, self).__init__()
        self.class_num = class_num
        #self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=256, class_num=class_num)
        # ---- ResNet Backbone ----
        self.pvt = TB()
        self.c = ChannelAttention()
        self.s = SpatialAttention()
        self.f = F()

        self.linear_pred = Conv2d(32, self.class_num, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # backbone1 resnet
        pvt = self.pvt(x)
        x1 = pvt[0]
        # print(t0.shape)
        x2 = pvt[1]
        # print(t1.shape)
        x3 = pvt[2]
        # print(t2.shape)
        x4 = pvt[3]
        # print(t3.shape)

        z1, z2, z3, z4 = self.c(x1, x2, x3, x4)
        #print(z1.shape, z2.shape, z3.shape, z4.shape)
        #s1, s2, s3, s4 = self.s(z1, z2, z3, z4)
        #print(s1.shape, s2.shape, s3.shape, s4.shape)
        f1 = self.f(z1, z2, z3, z4)
        #f1 = self.f(z1, z2, z3, z4)
        #print(f1.shape)

        p = self.dropout(f1)
        p = self.linear_pred(p)
        up = UpsamplingBilinear2d(scale_factor=4)
        features = up(p)
        return features

    # def _init_weights(self):
    #     #pretrained_dict = torch.load('/home/ge/ssformer/models/mit/mit_b2.pth')
    #     model_dict = self.backbone.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict)
    #     self.backbone.load_state_dict(model_dict)
    #     print("successfully loaded!!!!")

# if __name__=='__main__':
#     img = torch.randn(1, 3, 352, 352)
#
#     t = Tnet(
#         class_num=1,
#     )
#
#     preds = t(img)  # (1,1000)
#
#     print(preds.shape) #1,1,352,352

if __name__ == '__main__':

    model = Tnet().to('cuda')
    #from torchinfo import summary
#    summary(model, (1, 3, 352, 352))
    from thop import profile
    import torch
    input = torch.randn(1, 3, 352, 352).to('cuda')
    macs, params = profile(model, inputs=(input,))
    res = model(input)
    print(res.shape)
    # print(res[1].shape)
    print('macs:', macs / 1000000000)
    print('params:', params / 1000000)