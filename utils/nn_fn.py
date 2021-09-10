import torch
from torch import nn
import torch.nn.functional as F

class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
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

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=label.device)
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
    
    
class ClasModel(nn.Module):
    def __init__(self,in_features,out_features=10,n_layers=3,latent_features=3,mode="arc"):
        super().__init__()
        HIDDEN=64
        self.conv_0=nn.modules.Conv2d(in_features,HIDDEN,3,1,padding=(1,1),padding_mode='replicate')
        self.dropout_0=nn.modules.Dropout(0.2)
        
        self.convs = nn.ModuleList(
              [nn.modules.Conv2d(HIDDEN,HIDDEN,3,1,padding=(1,1),padding_mode='replicate') for _ in range(n_layers)])
        self.mxpools = nn.ModuleList([nn.modules.MaxPool2d(2,2) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.modules.Dropout(0.2) for _ in range(n_layers)])
        
        WIDTH=2
        self.gap=nn.AdaptiveAvgPool2d(WIDTH)
        self.flatten=nn.Flatten()
        self.dense=nn.modules.Linear(HIDDEN*WIDTH*WIDTH,latent_features)
        self.mode=mode
            
        self.out=nn.modules.Linear(latent_features,out_features)
        self.softmax=AddMarginProduct(latent_features,out_features,s=30,m=0.2)
        
    def forward(self,data,label=None):
        hiddens=[self.dropout_0(F.relu(self.conv_0(data)))]
        for cnv,mxp,drp in zip(self.convs,self.mxpools,self.dropouts):
            x=F.relu(cnv(hiddens[-1]))
            x=mxp(x)
            hiddens.append(drp(F.group_norm(x,4)))
        hiddens.append(self.flatten(self.gap(hiddens[-1])))
        hiddens.append(self.dense(hiddens[-1]))
        if label is None:
            return hiddens
        
        if self.mode=="arc":
            pred=self.softmax(hiddens[-1],label)
        else:
            pred=self.out(hiddens[-1])
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
def mean_acc(pred,y):
    return (pred.argmax(-1)==y).type(torch.float32).mean().item()