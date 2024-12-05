import numpy as np
np.random.seed(20)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
import sys
sys.path.append('../utilities')
from matplotlib import cm
import matplotlib.patches as mpatches
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array
import pandas as pd



# Drawings of the model predictions

def draw_model(mf_model, xmin, xmax, labels, factor=1., version='v1', x_fixed=[160,5,40,45,50]):
    SPLIT = 100
    ncol=3
    nrow=int(np.ceil(len(labels)/ncol))
    fig,ax = plt.subplots(nrow,ncol,figsize=(15, 5),constrained_layout=True)
    ax = fig.axes
    pdf=PdfPages(f'out/{version}/neutron-moderator-multi-fidelity-model_{version}.pdf')

    indices = [i for i in range(len(ax))]
    indices[0], indices[ncol-1] = indices[ncol-1], indices[0]

    for i in range(0,len(labels)):   
        
        ## Compute mean and variance predictions
        x_plot=[x_fixed[:] for l in range(0,SPLIT)]
        x_tmp = np.linspace(xmin[i], xmax[i], SPLIT)
        for k in range(0,SPLIT):
            x_plot[k][i]=x_tmp[k]
        x_plot = (np.atleast_2d(x_plot))
        X_plot = convert_x_list_to_array([x_plot , x_plot, x_plot])

        lf_mean_mf_model, lf_var_mf_model = mf_model.predict(X_plot[:SPLIT])
        lf_std_mf_model = np.sqrt(lf_var_mf_model)

        hf_mean_mf_model, hf_var_mf_model = mf_model.predict(X_plot[SPLIT:2*SPLIT])
        hf_std_mf_model = np.sqrt(hf_var_mf_model)

        hhf_mean_mf_model, hhf_var_mf_model = mf_model.predict(X_plot[2*SPLIT:])
        hhf_std_mf_model = np.sqrt(hhf_var_mf_model)

        ## Plot posterior mean and variance of nonlinear multi-fidelity model

        ax[indices[i]].fill_between(x_tmp.flatten(), (lf_mean_mf_model - factor * lf_std_mf_model).flatten(), 
                        (lf_mean_mf_model + factor * lf_std_mf_model).flatten(), color='darkturquoise', alpha=0.1)
        ax[indices[i]].plot(x_tmp, lf_mean_mf_model, '--', color='lightseagreen')
        ax[indices[i]].fill_between(x_tmp.flatten(), (hf_mean_mf_model - hf_std_mf_model).flatten(), 
                        (hf_mean_mf_model + hf_std_mf_model).flatten(), color='cadetblue', alpha=0.2)
        ax[indices[i]].plot(x_tmp, hf_mean_mf_model, '--', color='teal')
        ax[indices[i]].set_xlim(xmin[i], xmax[i])
        ax[indices[i]].set_ylabel(r'$y_{CNP}$', color="teal", fontsize=10)
        ax[indices[i]].tick_params(axis='y', labelcolor="teal")
        ax[indices[i]].set_xlabel(labels[i], fontsize=10)

        ax[indices[i]] = ax[indices[i]].twinx()
        ax[indices[i]].fill_between(x_tmp.flatten(), (hhf_mean_mf_model - hhf_std_mf_model).flatten(), 
                        (hhf_mean_mf_model + hhf_std_mf_model).flatten(), color='coral', alpha=0.4)    
        ax[indices[i]].plot(x_tmp, hhf_mean_mf_model, '--', color='orangered')

        ax[indices[i]].set_xlabel(labels[i], fontsize=10)
        ax[indices[i]].set_ylabel(r'$y_{raw}$', color="orangered", fontsize=10)
        ax[indices[i]].tick_params(axis='y', labelcolor="orangered")
        ax[indices[i]].set_xlim(xmin[i], xmax[i])
        
    for i in range(len(labels),len(ax)): 
        ax[i].set_axis_off()
    pdf.savefig(fig)
    pdf.close()


def draw_model_updated(fig, mf_model,xmin, xmax, labels, factor=1., version='v1', x_fixed=[160,5,40,45,50]):
    SPLIT = 100
    ax = fig.axes
    #fig.suptitle(r"Projection to x(r,b,N,$\theta$,L) ="+f"{x_fixed}")
    pdf=PdfPages(f'out/{version}/updated-neutron-moderator-multi-fidelity-model_{version}.pdf')

    for i in range(0,len(labels)):   

        ## Compute mean and variance predictions
        x_plot=[x_fixed[:] for l in range(0,SPLIT)]
        x_tmp = np.linspace(xmin[i], xmax[i], SPLIT)
        for k in range(0,SPLIT):
            x_plot[k][i]=x_tmp[k]
        x_plot = (np.atleast_2d(x_plot))
        X_plot = convert_x_list_to_array([x_plot , x_plot, x_plot])

        lf_mean_mf_model, lf_var_mf_model = mf_model.predict(X_plot[:SPLIT])
        lf_std_mf_model = np.sqrt(lf_var_mf_model)

        hf_mean_mf_model, hf_var_mf_model = mf_model.predict(X_plot[SPLIT:2*SPLIT])
        hf_std_mf_model = np.sqrt(hf_var_mf_model)

        hhf_mean_mf_model, hhf_var_mf_model = mf_model.predict(X_plot[2*SPLIT:])
        hhf_std_mf_model = np.sqrt(hhf_var_mf_model)

        ## Plot posterior mean and variance of nonlinear multi-fidelity model

        ax[i].fill_between(x_tmp.flatten(), (lf_mean_mf_model - factor * lf_std_mf_model).flatten(), 
                        (lf_mean_mf_model + factor * lf_std_mf_model).flatten(), color='darkturquoise', alpha=0.1)
        ax[i].plot(x_tmp, lf_mean_mf_model, '--', color='lightseagreen')
        ax[i].fill_between(x_tmp.flatten(), (hf_mean_mf_model - hf_std_mf_model).flatten(), 
                        (hf_mean_mf_model + hf_std_mf_model).flatten(), color='cadetblue', alpha=0.2)
        ax[i].plot(x_tmp, hf_mean_mf_model, '--', color='teal')
        ax[i].set_xlim(xmin[i], xmax[i])
        ax[i].set_ylabel(r'$y_{CNP}$', color="teal", fontsize=10)
        ax[i].tick_params(axis='y', labelcolor="teal")
        ax[i].set_xlabel(labels[i], fontsize=10)

        ax[i] = ax[i].twinx()
        ax[i].fill_between(x_tmp.flatten(), (hhf_mean_mf_model - hhf_std_mf_model).flatten(),(hhf_mean_mf_model + hhf_std_mf_model).flatten(), color='coral', alpha=0.3)    
        ax[i].plot(x_tmp, hhf_mean_mf_model, '--', color='orangered')

        ax[i].set_xlabel(labels[i], fontsize=10)
        ax[i].set_ylabel(r'$y_{raw}$', color="orangered", fontsize=10)
        ax[i].tick_params(axis='y', labelcolor="orangered")
        ax[i].set_xlim(xmin[i], xmax[i])
        
    for i in range(len(labels),len(ax)): 
        ax[i].set_axis_off()
    pdf.savefig(fig)
    pdf.close()
    return fig

# Drawings of the aquisition function

def draw_acquisition_func(fig, us_acquisition, xlow, xhigh, labels, x_next=np.array([]), version='v1', x_fixed=[160,5,40,45,50]):
    SPLIT = 50
    df= pd.DataFrame()
    ax2 = fig.axes
    pdf=PdfPages(f'out/{version}/acquisition-function_{version}.pdf')
    
    for i in range(0,len(xlow)):
        ax2[i].set_title(f'Projected acquisition function - {labels[i]}');
        
        x_plot=[x_fixed[:] for l in range(0,SPLIT)]
        x_tmp = np.linspace(xlow[i], xhigh[i], SPLIT)
        for k in range(0,SPLIT):
            x_plot[k][i]=x_tmp[k]
        x_plot = (np.atleast_2d(x_plot))
        X_plot = convert_x_list_to_array([x_plot , x_plot])
        acq=us_acquisition.evaluate(X_plot[SPLIT:])
        color = next(ax2[i]._get_lines.prop_cycler)['color']
        ax2[i].plot(x_tmp,acq/acq.max(),color=color)
        acq=us_acquisition.evaluate(X_plot[:SPLIT])
        ax2[i].plot(x_tmp,acq/acq.max(),color=color,linestyle="--")
        
        #x_plot = np.linspace(xlow[i], xhigh[i], 500)[:, None]
        #x_plot_low = np.concatenate([np.atleast_2d(x_plot), np.zeros((x_plot.shape[0], 1))], axis=1)
        #x_plot_high = np.concatenate([np.atleast_2d(x_plot), np.ones((x_plot.shape[0], 1))], axis=1)
        #print(us_acquisition.evaluate(x_plot_low))
        #ax2[i].plot(x_plot_low[:, 0], us_acquisition.evaluate(x_plot_low), 'b')
        #ax2[i].plot(x_plot_high[:, 0], us_acquisition.evaluate(x_plot_high), 'r')
        
        if x_next.any():
            ax2[i].axvline(x_next[0,i], color="red", label="x_next", linestyle="--")
            ax2[i].text(x_next[0,i]+0.5,0.95,f'x = {round(x_next[0,i],1)}', color='red', fontsize=8)

        ax2[i].set_xlabel(f"{labels[i]}")
        ax2[i].set_ylabel(r"$\mathcal{I}(x)$")
        
        #for j in range(int(len(ax2[i].lines)/2)):
        #    df[f'x{j}_{labels[i]}']=np.array(np.round(ax2[i].lines[2*j].get_xdata(),3))
        #    df[f'y{j}_{labels[i]}']=np.array(np.round(ax2[i].lines[2*j].get_ydata(),3))

    pdf.savefig(fig)
    pdf.close()
    df.to_csv(f'out/{version}/acquisition-function_{version}.csv')
    return fig
       
def draw_model_acquisition_func(fig1, fig2, labels, version='v1'):
    
    pdf=PdfPages(f'out/{version}/model-acquisition-evolution_{version}.pdf')
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
        pdf.savefig(fig5)
    
    pdf.close()
    plt.show()




def read_acquisition_function(filename, labels):
    data=pd.read_csv(filename,index_col=0)
    nsamples=int(len(data.columns)/(2*len(labels)))

    fig = [plt.figure(figsize=(12,5)) for i in range(len(labels))]
    for idx,l in enumerate(labels):
        ax = fig[idx].gca()
        for i in range(nsamples):
            x=data[f'x{i}_{l}'].to_numpy()
            y=data[f'y{i}_{l}'].to_numpy()
            ax.plot(x,y)
            ax.set_xlabel(l)
            ax.set_ylim(0,1.1)

def model_validation(mf_model, file_in, x_labels, y_label, version):
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
                hhf_mean_mf_model, hhf_var_mf_model = mf_model.predict(X_plot[2*SPLIT:])
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
        plt.savefig(f'out/{version}/model-validation_{version}.pdf')
        return fig