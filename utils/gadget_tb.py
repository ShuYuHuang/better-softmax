
import matplotlib.pyplot as plt
import IPython.display as dspl
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter


import torch
import numpy as np

def mean_acc(pred,y):
    return (pred.argmax(-1)==y).type(torch.float32).mean().item()
def show_progress(idx,**args):
    if idx>1:
        dspl.update_display(f"{[arg+f':{args[arg]:.04f}' if isinstance(args[arg],float) else arg+f':{args[arg]}' for arg in args ]}",display_id=1)

        
class TBWriter(SummaryWriter):
    step=0
    def __init__(self,location):
        super().__init__(location)

    def plt_once(self,**args):
        self.step+=1
        for arg in args:
            if "loss" in arg:
                self.add_scalars('run_arc/loss',{arg: args[arg]},self.step)
                continue
            if "acc" in arg:
                self.add_scalars('run_arc/acc',{arg: args[arg]},self.step)
                continue
            if "weight" in arg:
                self.add_scalars('run_arc/weight',{arg: args[arg]},self.step)
                continue
            if "latent" in arg:
                self.add_scalars('run_arc/latent',{arg: args[arg]},self.step)

def to_numpy(x: torch.Tensor)->np.array:
    return x.detach().cpu().numpy()

def plt_latent_dist(latent,y):
    fig = plt.figure(1, figsize=(10, 9))
    ax = Axes3D(fig, elev=48, azim=134)
    ax.scatter(to_numpy(latent[:, 0]),
               to_numpy(latent[:, 1]),
               to_numpy(latent[:, 2]),
               c=to_numpy(y),
               cmap=plt.cm.Set1, edgecolor='k')

    for label in range(10):
        ax.text3D(to_numpy(latent[y == label, 0]).mean(),
                  to_numpy(latent[y == label, 1]).mean(),
                  to_numpy(latent[y == label, 2]).mean(), f"{label}",
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'),size=25)

    ax.set_title("3D visualization", fontsize=20)
    ax.set_xlabel("latent1", fontsize=16)
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("latent2", fontsize=16)
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("latent3", fontsize=16)
    ax.w_zaxis.set_ticklabels([])
    plt.show()

