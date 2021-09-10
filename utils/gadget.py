
import matplotlib.pyplot as plt
import IPython.display as dspl



def plt_once(ee,idx,**args):
    total=len(args)
    colors=("--r","--g","--b","--k")
    if idx>1:
        for n,arg in enumerate(args):
            plt.subplot(1,total,n+1)
            if idx>100:
                grid=int(idx//100)
                plt.plot(list(range(0,idx+1,grid)),args[arg][::grid],colors[n%4])
            else:
                plt.plot(list(range(idx+1)),args[arg],colors[n%4])
            
            plt.title(arg)
            plt.xlabel("batch")
            plt.ylabel("amount")
        dspl.update_display(plt.gcf(),display_id=2)
        plt.cla()
        
def show_progress(ee,idx,**args):
    if idx>1:
        dspl.update_display(f"epoch:{ee},{[arg+f':{args[arg][-1]:.04f}' for arg in args]}",display_id=1)