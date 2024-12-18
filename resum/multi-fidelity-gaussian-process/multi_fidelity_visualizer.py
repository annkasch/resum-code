import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array
import pandas as pd
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
resum_path = os.getenv("RESUM_PATH")

class MultiFidelityVisualizer():
    def __init__(self, mf_model, parameters, x_fixed):
         mf_model = mf_model
         parameters=parameters
         x_fixed=x_fixed

    # Drawings of the model predictions projecting each dimension on a fixed point in space for the remaining dimensions
    def draw_model(self, fig):
        SPLIT = 100
        ncol=3

        #fig,ax = plt.subplots(nrow,ncol,figsize=(15, 5),constrained_layout=True)
        ax = fig.axes

        indices = [i for i in range(len(ax))]
        indices[0], indices[ncol-1] = indices[ncol-1], indices[0]
        colors_std = ['darkturquoise','cadetblue','coral']
        colors_mean = ['lightseagreen','teal','orangered']
        for i in range(len(self.parameters)):   
            
            ## Compute mean and variance predictions
            x_plot=[self.x_fixed[:] for l in range(0,SPLIT)]
            x_tmp = np.linspace(self.parametes[i][0], self.parametes[i][1], SPLIT)
            for k in range(0,SPLIT):
                x_plot[k][i]=x_tmp[k]
            x_plot = (np.atleast_2d(x_plot))
            X_plot = convert_x_list_to_array([x_plot , x_plot, x_plot])

            for f in range(self.mf_model.nfidelities):
                f_mean_mf_model, f_var_mf_model = self.mf_model.model.predict(X_plot[f*SPLIT:(f+1)*SPLIT])
                f_std_mf_model = np.sqrt(f_var_mf_model)

                ax[indices[i]].fill_between(x_tmp.flatten(), (f_mean_mf_model - f_std_mf_model).flatten(), 
                            (f_mean_mf_model + f_std_mf_model).flatten(), color=colors_std[f], alpha=0.1)
                ax[indices[i]].plot(x_tmp,f_mean_mf_model, '--', color=colors_mean[f])

            ax[indices[i]].set_xlabel(self.parameters[i], fontsize=10)
            ax[indices[i]].set_ylabel(r'$y_{raw}$')
            ax[indices[i]].set_xlim(self.parametes[i][0], self.parametes[i][1])
            
        for i in range(len(self.parameters),len(ax)): 
            ax[i].set_axis_off()
        return fig

    # Drawings of the aquisition function
    def draw_acquisition_func(self, fig, us_acquisition, x_next=np.array([])):
        SPLIT = 50
        ax2 = fig.axes
        
        for i in range(len(self.parameters)):
            ax2[i].set_title(f'Projected acquisition function - {self.parameters[i]}');
            
            x_plot=[self.x_fixed[:] for l in range(0,SPLIT)]
            x_tmp = np.linspace(self.parametes[i][0], self.parametes[i][1], SPLIT)
            for k in range(0,SPLIT):
                x_plot[k][i]=x_tmp[k]
            x_plot = (np.atleast_2d(x_plot))
            X_plot = convert_x_list_to_array([x_plot , x_plot])
            acq=us_acquisition.evaluate(X_plot[SPLIT:])
            color = next(ax2[i]._get_lines.prop_cycler)['color']
            ax2[i].plot(x_tmp,acq/acq.max(),color=color)
            acq=us_acquisition.evaluate(X_plot[:SPLIT])
            ax2[i].plot(x_tmp,acq/acq.max(),color=color,linestyle="--")
            
            if x_next.any():
                ax2[i].axvline(x_next[0,i], color="red", label="x_next", linestyle="--")
                ax2[i].text(x_next[0,i]+0.5,0.95,f'x = {round(x_next[0,i],1)}', color='red', fontsize=8)

            ax2[i].set_xlabel(f"{self.parameters[i]}")
            ax2[i].set_ylabel(r"$\mathcal{I}(x)$")

        return fig

    def model_validation(self, file_in, x_labels, y_label, version):
            data=pd.read_csv(file_in)
            #data=data[[f'Mode', x_labels[0], x_labels[1], x_labels[2], x_labels[3], x_labels[4],y_label]]

            x_train_hf_sim = data.loc[data['Mode']==1.][x_labels].to_numpy().tolist()
            y_train_hf_sim = data.loc[data['Mode']==1.][y_label].to_numpy().tolist()
            x_train_hf_sim, y_train_hf_sim = (np.atleast_2d(x_train_hf_sim), np.atleast_2d(y_train_hf_sim).T)

            counter_1sigma = 0
            counter_2sigma = 0
            counter_3sigma = 0
            nsamples = 0
            mfsm_model_mean = np.empty(shape=[0, 0])
            mfsm_model_std = np.empty(shape=[0, 0])
            hf_data=[]
            x=[]
            for i in range(len(x_train_hf_sim)-1):
                    nsamples += 1
                    SPLIT = 1
                    x_plot = (np.atleast_2d(x_train_hf_sim[i]))
                    X_plot = convert_x_list_to_array([x_plot , x_plot, x_plot])
                    hhf_mean_mf_model, hhf_var_mf_model = self.mf_model.predict(X_plot[2*SPLIT:])
                    hhf_std_mf_model = np.sqrt(hhf_var_mf_model)

                    hf_data.append(y_train_hf_sim[i])
                    x.append(i)
                    mfsm_model_mean=np.append(mfsm_model_mean,hhf_mean_mf_model[0,0])
                    mfsm_model_std=np.append(mfsm_model_std,hhf_std_mf_model[0,0])
                    if (y_train_hf_sim[i] < hhf_mean_mf_model+hhf_std_mf_model) and (y_train_hf_sim[i] > hhf_mean_mf_model-hhf_std_mf_model):
                            counter_1sigma += 1
                    if (y_train_hf_sim[i] < hhf_mean_mf_model+2*hhf_std_mf_model) and (y_train_hf_sim[i] > hhf_mean_mf_model-2*hhf_std_mf_model):
                            counter_2sigma += 1
                    if (y_train_hf_sim[i] < hhf_mean_mf_model+3*hhf_std_mf_model) and (y_train_hf_sim[i] > hhf_mean_mf_model-3*hhf_std_mf_model):
                            counter_3sigma += 1

            print("1 sigma: ", counter_1sigma/nsamples*100.," %" )
            print("2 sigma: ", counter_2sigma/nsamples*100.," %" )
            print("3 sigma: ", counter_3sigma/nsamples*100.," %" )

            fig = plt.subplots(figsize=(12, 2.5))
            #plt.bar(x=np.arange(len(mfsm_model_mean)), height=mfsm_model_mean, color="lightgray", label='RESuM')
            plt.fill_between(x=np.arange(len(mfsm_model_mean)), y1=mfsm_model_mean-3*mfsm_model_std, y2=mfsm_model_mean+3*mfsm_model_std, color="coral",alpha=0.2, label=r'$\pm 3\sigma$')
            plt.fill_between(x=np.arange(len(mfsm_model_mean)), y1=mfsm_model_mean-2*mfsm_model_std, y2=mfsm_model_mean+2*mfsm_model_std, color="yellow",alpha=0.2, label=r'$\pm 2\sigma$')
            plt.fill_between(x=np.arange(len(mfsm_model_mean)), y1=mfsm_model_mean-mfsm_model_std, y2=mfsm_model_mean+mfsm_model_std, color="green",alpha=0.2, label=r'RESuM $\pm 1\sigma$')
            plt.xlabel('HF Simulation Trial Number')
            plt.ylim(0.,0.55)
            plt.ylabel(r'$y_{raw}$')
            plt.plot(x[:],hf_data[:],'.',color="black", label="HF Validation Data")
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [3,2,1,0]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=9, bbox_to_anchor=(0.665,1.),ncol=5)
            plt.savefig(f'{resum_path}/out/{version}/model-validation_{version}.pdf')
            return fig
    
def draw_model_acquisition_func(fig1, fig2, labels):

    ax1 = fig1.axes
    ax2 = fig2.axes

    for index, label in enumerate(labels):
        ncurves_per_fig_update=3
        nsamples=int((len(ax1[index].lines))/ncurves_per_fig_update)
        
        fig5,(ax20,ax21,ax22,ax23)=plt.subplots(nrows=4,ncols=1,sharex=True, gridspec_kw={'height_ratios': [1,1,1, 1]})
        ax20.set_ylabel('$y_{CNP}$')
        ax21.set_ylabel('$y_{CNP}$')
        ax22.set_ylabel('$y_{raw}$')
        ax23.set_ylabel(r'$\mathcal{I}(x)$')
        ax23.set_xlabel(label)

        for i in range(nsamples+1):
            
            idx=2*(i+1)-1
            #print(index, i, [l.get_alpha() for l in ax1[index].collections], len(ax1))
            for j in range(idx+1):
                poly=ax1[index].collections[j]
                x1=poly.get_paths()[0].vertices
                x1s=x1[:,0]
                split=int((len(x1s)-1)/2)
                x1s=x1s[:split]
                y1s=x1[:split,1]
                x2=poly.get_paths()[0].vertices
                y2s=x2[split+1:,1]
                y2s=y2s[::-1]

                if ax1[index].collections[j].get_alpha()==0.1:
                    ax20.fill_between(x1s.flatten(),y1s.flatten(),y2s.flatten(),color='cadetblue', alpha=0.1)
                elif ax1[index].collections[j].get_alpha()==0.2:
                    ax21.fill_between(x1s.flatten(),y1s.flatten(),y2s.flatten(),color='darkturquoise', alpha=0.2)

            curve0=ax1[index].lines[idx-1]
            ax20.plot(curve0.get_xdata(),curve0.get_ydata(),'--',color='teal')
            curve0=ax1[index].lines[idx-2]
            ax21.plot(curve0.get_xdata(),curve0.get_ydata(),'--',color='lightseagreen')
            
            index2=len(labels)+1+len(labels)*i+index

            poly=ax1[index2].collections[0]            
            x1=poly.get_paths()[0].vertices
            x1s=x1[:,0]
            split=int((len(x1s)-1)/2)
            x1s=x1s[:split]
            y1s=x1[:split,1]
            x2=poly.get_paths()[0].vertices
            y2s=x2[split+1:,1]
            y2s=y2s[::-1]

            if ax1[index2].collections[0].get_alpha()==0.3:
                ax22.fill_between(x1s.flatten(),y2s.flatten(),y1s.flatten(),color='orangered', alpha=0.3)
            
            curve0=ax1[index2].lines[0]
            ax22.plot(curve0.get_xdata(),curve0.get_ydata(),'--',color='orangered')

            if len(ax2[index].lines) > 0 :
                color = next(ax21._get_lines.prop_cycler)['color']
                curve2=ax2[index].lines[3*i+2]
                #ax21.plot(curve2.get_xdata(),curve2.get_ydata(), color="lightgray", linestyle="--")

                curve1=ax2[index].lines[3*i]
                ax23.plot(curve1.get_xdata(),curve1.get_ydata(), color=color, label=f"sample #{i} HF (-- LF)")
                curve2=ax2[index].lines[3*i+1]
                ax23.plot(curve2.get_xdata(),curve2.get_ydata(), color=color, linestyle="--")

        fig5.subplots_adjust(hspace=0.1)
    plt.show()