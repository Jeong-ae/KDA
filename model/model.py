import math
from torch.nn import functional as F
from .backbone import *


class AttenHead(nn.Module):
    def __init__(self, fdim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.fatt = fdim//num_heads # 128 / 4 = 32

        for i in range(num_heads):
            setattr(self, f'embd{i}', nn.Linear(fdim, self.fatt)) # (128, 32) self.embd0 ... self.embd3
        for i in range(num_heads):
            setattr(self, f'fc{i}', nn.Linear(2*self.fatt, self.fatt)) # (64, 32) self.fc0 ... self.fc3
        self.fc = nn.Linear(self.fatt*num_heads, fdim) # (128, 128)
        self.dropout = nn.Dropout(0.1)

    def forward(self, fx_in, fp_in):
        fp_in = fp_in.squeeze(0) # 첫번째 차원의 크기가 1이 아니면 아무일도 안일어남
        d = math.sqrt(self.fatt) # 5.6568... attention scaling

        Nx = len(fx_in) # batch size??
        f = torch.cat([fx_in, fp_in]) # (b+10, 128)
        f = torch.stack([getattr(self, f'embd{i}')(f) for i in range(self.num_heads)])  # head x N x fatt
        # linear function에 f(2차원 tensor)를 input으로 넣음 
        fx, fp = f[:, :Nx], f[:, Nx:] # fx: (4,b,32) fp: (4,10,32)

        # attention calculation
        w = self.dropout(F.softmax(torch.matmul(fx, torch.transpose(fp, 1, 2)) / d, dim=2))  # head x Nx x Np (4,b,10)
        fa = torch.cat([torch.matmul(w, fp), fx], dim=2)  # head x Nx x 2*fatt # 세번째 차원으로 합쳐짐
        fa = torch.stack([F.relu(getattr(self, f'fc{i}')(fa[i])) for i in range(self.num_heads)])  # head x Nx x fatt
        fa = torch.transpose(fa, 0, 1).reshape(Nx, -1)  # Nx x fdim
        fx = F.relu(fx_in + self.fc(fa))  # Nx x fdim
        w = torch.transpose(w, 0, 1)  # Nx x head x Np

        return fx, w


class FeatMatch(nn.Module): # Student
    def __init__(self, backbone, num_classes, devices,  pretrain, num_heads=1, amp=True):
        super().__init__()
        self.mode = 'train'
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.devices = devices
        self.default_device = torch.device('cuda', devices[0]) if devices is not None else torch.device('cpu')
        fext, self.fdim = make_backbone(backbone) # (, 512)
        #if backbone != 'vit':
        #self.fext = nn.DataParallel(AmpModel(fext, amp), devices)
        self.fext = fext
       # else : self.fext = fext
        if not pretrain:
            self.adapt = 192 #(티쳐가 CNN이면 128 VIT면 192 RESNET이면 512)
            self.Lin = nn.Linear(128,self.adapt)
            self.atten = AttenHead(self.adapt, num_heads)
            self.clf = nn.Linear(self.adapt, num_classes)
        else:
            self.Lin = nn.Identity()
            self.atten = AttenHead(self.fdim, num_heads)
            self.clf = nn.Linear(self.fdim, num_classes)
        self.backbone=backbone

    def set_mode(self, mode):
        self.mode = mode

    def extract_feature(self, x):
        if self.backbone == 'vit':
            x = self.fext.forward_features(x) # (batch, 197, 192)
            x = x[:,0,:].reshape(-1, 192) #(batch, 192)
            return x
        else:
            x = self.fext(x) # feature extractor , (batch, 196, 192) -> (batch, 192)
            x = self.Lin(x) #linear
            return x

    def forward(self, x, fp=None):
        if self.mode == 'fext':
            return self.extract_feature(x)

        elif self.mode == 'pretrain': # 얘는 feature를 뽑고, clf 함
            fx = self.extract_feature(x) # (192, 512)
            cls_x = self.clf(fx) # (192, 10)

            return cls_x

        elif self.mode == 'train': # 얘는 feature를 뽑고, attention module을 거치고 clf 함
            fx = self.extract_feature(x)
            if self.devices is not None:
                inputs = (fx, fp.unsqueeze(0).repeat(len(self.devices), 1, 1))
                fxg, wx = nn.parallel.data_parallel(self.atten, inputs, device_ids=self.devices)
            else:
                fxg, wx = self.atten(fx, fp.unsqueeze(0))

            cls_xf = self.clf(fx) # (b, 10) FA 통과안한놈
            cls_xg = self.clf(fxg) # (b, 10) FA 통과한놈
            # return 
            return cls_xg, cls_xf, fx, fxg, wx

        else:
            raise ValueError
