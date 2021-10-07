
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("Set2"))

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
import datetime

def plot_2_lines(datax,pred,cut_num,text='',save_image=True):
    header=f"Prediction Model and Series Name: {text}"
    rts=pd.DataFrame(datax,columns=['Actual'])
    rts['Forecast']=pred.flatten()
    rts['timepoint']=np.arange(len(rts))
    d=rts.melt(id_vars=['timepoint'])

    fig, ax4 = plt.subplots(figsize=(15,6))
    ax4=sns.lineplot(data=d,x='timepoint',y='value',hue='variable',style="variable",markers=True,linewidth = 2,markersize=10)
    ax4.axvline(x=cut_num, c='gray', linestyle='--')

    ax4.set_title(header,fontsize=18,weight='bold')
    leg=ax4.get_legend().set_title('Line')
    ax4.set_xlabel('X_axis', fontsize=15);
    ax4.set_ylabel('Y_axis', fontsize=15);

    dir_path=f"graphs"
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    if save_image:
        datestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename=f"graphs/2lineplot_{text.replace(' ','_').replace('/','_')}_{datestamp}.png"
        fig.savefig(filename)
    return fig


# show fitted curves:
def plot_curve(datax, pred, cut_num):
    plt.axvline(x=cut_num, c='r', linestyle='--')
    plt.plot(np.array(datax))
    plt.plot(np.array(pred))
    plt.suptitle('Time-Series Prediction')
    plt.show()





def plot_simple_curve(df):
    return plt.plot(df)