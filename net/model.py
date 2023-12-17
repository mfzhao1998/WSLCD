import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np

class Net_sig(nn.Module):
    def __init__(self, n_classes=1,output_stride=8, backbone='resnet18', multiscale='multiscale'):
        super(Net_sig, self).__init__()
        self.backbone=timm.create_model(backbone, pretrained=False,features_only= True , output_stride= output_stride , out_indices=( 2 , 3,  4 ))
        self.multiscale=multiscale
        if backbone=='resnet18' or backbone=='resnet34':
            if self.multiscale=='multiscale':
                self.fn=512+256+128
            else:
                self.fn=512
        elif backbone=='resnet50':
            if self.multiscale=='multiscale':
                self.fn=2048+1024+512
            else:
                self.fn=2048
        self.classifier = nn.Conv2d(self.fn, 1, 1, bias=False)
        self.fc_proj = torch.nn.Conv2d(self.fn, 128, 1, bias=False)
        self.n_classes = n_classes
        self.sig = nn.Sigmoid()
        self.dp=nn.Dropout(0.5)
        self.out_stride=output_stride

    def forward(self, x1,x2):
        N, C, H, W = x1.size()
        x12, x13, x14 = self.backbone(x1)
        x22, x23, x24 = self.backbone(x2)
        if self.multiscale=='multiscale':
            if self.out_stride==8:
                cam = torch.cat((torch.abs(x12-x22),torch.abs(x13-x23),torch.abs(x14-x24)), 1)
            else:
                cam = torch.cat((torch.abs(x12-x22),F.interpolate(torch.abs(x13-x23), size=x12.size()[2:], mode='bilinear', align_corners=True),F.interpolate(torch.abs(x14-x24), size=x12.size()[2:], mode='bilinear', align_corners=True)), 1)
            
        else:
            cam = torch.abs(x14-x24)
        f_proj = F.relu(self.fc_proj(cam), inplace=True)
        cam = self.classifier(cam)
        cam = F.interpolate(cam, (H,W), mode='bilinear', align_corners=True)
        
        return cam, f_proj


