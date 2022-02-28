import torch
from torch import nn
import torch.nn.functional as F
import math

class AddMarginProduct(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=15.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine

        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=label.device)
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        loss = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        loss *= self.s
        # print(output)

        return cosine,loss

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

class ArcMarginProduct(nn.Module):
    def __init__(self, in_feature, out_feature, s=32.0, m=0.40, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight,gain=1.0)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.eps = torch.tensor(1e-10,dtype=torch.float32)
        
    def forward(self, x, label=None):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if label is None:
            return cosine

        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        loss = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        loss *= self.s

        return cosine,loss

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
    
    
class ClasModel(nn.Module):
    def __init__(self,in_features,out_features=10,n_layers=3,latent_features=3,hidden_featrues=64,mode="cos",s=30,m=0.1, easy_margin=False):
        super().__init__()
        self.conv_0=nn.modules.Conv2d(in_features,hidden_featrues,3,1,padding=(1,1),padding_mode='replicate')
        self.dropout_0=nn.modules.Dropout(0.2)
        
        self.convs = nn.ModuleList(
              [nn.modules.Conv2d(hidden_featrues,hidden_featrues,3,1,padding=(1,1),padding_mode='replicate') for _ in range(n_layers)])
        self.mxpools = nn.ModuleList([nn.modules.MaxPool2d(2,2) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.modules.Dropout(0.2) for _ in range(n_layers)])
        
        WIDTH=2
        self.gap=nn.AdaptiveAvgPool2d(WIDTH)
        self.flatten=nn.Flatten()
        self.dense=nn.modules.Linear(hidden_featrues*WIDTH*WIDTH,latent_features)
        self.mode=mode
        if self.mode=="dense":
            self.out_net=nn.modules.Linear(latent_features,out_features)
        if self.mode=="cos":
            self.out_net=AddMarginProduct(latent_features,out_features,s=s,m=m)
        if self.mode=="arc":
            self.out_net=ArcMarginProduct(latent_features,out_features,s=s,m=m,easy_margin=easy_margin)
        
        
    def forward(self,data,label=None):
        hiddens=[self.dropout_0(F.relu(self.conv_0(data)))]
        for cnv,mxp,drp in zip(self.convs,self.mxpools,self.dropouts):
            x=F.relu(cnv(hiddens[-1]))
            x=mxp(x)
            hiddens.append(drp(F.group_norm(x,4,eps=1e-10)))
        hiddens.append(self.flatten(self.gap(hiddens[-1])))
        hiddens.append(self.dense(hiddens[-1]))
        
        
        if (label is None) or (self.mode=="dense"):
            pred=self.out_net(hiddens[-1])
        else:
            pred,loss=self.out_net(hiddens[-1],label)
            return pred,loss,hiddens

        return pred,hiddens
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-10):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = torch.tensor(eps,dtype=torch.float32)
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input+self.eps, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
    