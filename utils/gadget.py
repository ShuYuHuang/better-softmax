import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import plotly.express as px
import IPython.display as dspl



def mean_acc(pred,y):
    return (pred.argmax(-1)==y).type(torch.float32).mean().item()
def to_numpy(x: torch.Tensor)->np.array:
    return x.detach().cpu().numpy()
def show_progress(idx,display_id,**args):
    if idx>1:
        dspl.update_display(f"{[arg+f':{args[arg]:.04f}' if isinstance(args[arg],float) else arg+f':{args[arg]}' for arg in args ]}",display_id=display_id)


class Rec():
    def __init__(self):
        self.df=pd.DataFrame()
    def append(self,ltn,y,itr):
        # arc=F.normalize(ltn)
        df_new=pd.DataFrame()
        df_new["f1"]=to_numpy(ltn)[:,0]
        df_new["f2"]=to_numpy(ltn)[:,1]
        df_new["arc"]=0.5
        df_new["iterations"]=np.repeat(itr,len(ltn))
        df_new["label"]=to_numpy(y)
        self.df=self.df.append(df_new)
    def normalize(self):
        # Normalize latent by maximum value
        self.mx_norm=np.linalg.norm(self.df[["f1","f2"]].values,axis=1).max()
        self.df[["f1",'f2']]=self.df[["f1",'f2']].apply(lambda x: x/self.mx_norm)

        # Project latents into arc
        df=self.df.copy()
        norm=np.linalg.norm(df[["f1","f2"]].values,axis=1,keepdims=True)
        df[["f1",'f2']]=df[["f1",'f2']].values/norm
        df["arc"]=df["arc"].values/0.5*0.9

        self.df=self.df.append(df)


    def __repr__(self):
        fig=px.scatter(self.df, x="f1", y="f2", animation_frame="iterations", color="label",opacity=self.df.arc,
                range_x=[-1.5,1.5],
              range_y=[-1.5,1.5])
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        )
        return fig
