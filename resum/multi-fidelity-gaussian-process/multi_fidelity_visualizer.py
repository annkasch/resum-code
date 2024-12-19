import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array
from sklearn.metrics import mean_squared_error

class MultiFidelityVisualizer():
    def __init__(self, mf_model, parameters, x_fixed):
        self.mf_model = mf_model
        self.parameters = parameters
        self.x_fixed = x_fixed
        self.colors_std = ['darkturquoise','cadetblue','coral']
        self.colors_mean = ['lightseagreen','teal','orangered']

    # Drawings of the model predictions projecting each dimension on a fixed point in space for the remaining dimensions
    def draw_model_projections(self, fig):
        SPLIT = 100
        ncol=3

        #fig,ax = plt.subplots(nrow,ncol,figsize=(15, 5),constrained_layout=True)
        ax = fig.axes
        
        indices = [i for i in range(len(ax))]
        indices[0], indices[ncol-1] = indices[ncol-1], indices[0]

        for i,p in enumerate(self.parameters):   
            ## Compute mean and variance predictions
            x_plot=[self.x_fixed[:] for l in range(0,SPLIT)]
            x_tmp = np.linspace(self.parameters[p][0], self.parameters[p][1], SPLIT)
            for k in range(0,SPLIT):
                x_plot[k][i]=x_tmp[k]
            x_plot = (np.atleast_2d(x_plot))
            X_plot = convert_x_list_to_array([x_plot , x_plot, x_plot])

            for f in range(self.mf_model.nfidelities):
                f_mean_mf_model, f_var_mf_model = self.mf_model.model.predict(X_plot[f*SPLIT:(f+1)*SPLIT])
                f_std_mf_model = np.sqrt(f_var_mf_model)

                ax[indices[i]].fill_between(x_tmp.flatten(), (f_mean_mf_model - f_std_mf_model).flatten(), 
                            (f_mean_mf_model + f_std_mf_model).flatten(), color=self.colors_std[f], alpha=0.1)
                ax[indices[i]].plot(x_tmp,f_mean_mf_model, '--', color=self.colors_mean[f])

            ax[indices[i]].set_xlabel(p, fontsize=10)
            ax[indices[i]].set_ylabel(r'$y_{raw}$')
            ax[indices[i]].set_xlim(self.parameters[p][0], self.parameters[p][1])
            
        for i in range(len(self.parameters),len(ax)): 
            ax[i].set_axis_off()
        return fig

    # Drawings of the aquisition function
    def draw_acquisition_func(self, fig, us_acquisition, x_next=np.array([])):
        SPLIT = 50
        ax2 = fig.axes

        for i, p in enumerate(self.parameters):
            ax2[i].set_title(f"Projected acquisition function - {p}")
            x_plot = [self.x_fixed[:] for _ in range(SPLIT)]
            x_tmp = np.linspace(self.parameters[p][0], self.parameters[p][1], SPLIT)
            for k in range(SPLIT):
                x_plot[k][i] = x_tmp[k]
            x_plot = np.atleast_2d(x_plot)
            X_plot = convert_x_list_to_array([x_plot, x_plot])
            
            acq = us_acquisition.evaluate(X_plot[SPLIT:])
            try:
                color = next(ax2[i].get_prop_cycle())["color"]
            except AttributeError:
                color = "blue"  # Fallback color if cycle is unavailable
            
            ax2[i].plot(x_tmp, acq / acq.max(), color=color)
            
            acq = us_acquisition.evaluate(X_plot[:SPLIT])
            ax2[i].plot(x_tmp, acq / acq.max(), color=color, linestyle="--")
            
            if x_next.any():
                ax2[i].axvline(x_next[0, i], color="red", label="x_next", linestyle="--")
                ax2[i].text(
                    x_next[0, i] + 0.5, 0.95,
                    f"x = {round(x_next[0, i], 1)}",
                    color="red", fontsize=8
                )
            
            ax2[i].set_xlabel(p)
            ax2[i].set_ylabel(r"$\mathcal{I}(x)$")

        return fig

    def model_validation(self, x_test, y_test):
            x_test, y_test = (np.atleast_2d(x_test), np.atleast_2d(y_test).T)

            counter_1sigma = 0
            counter_2sigma = 0
            counter_3sigma = 0

            mfsm_model_mean = np.empty(shape=[0, 0])
            mfsm_model_std = np.empty(shape=[0, 0])
            y_data=[]
            x=[]
            for i in range(len(x_test)):

                    SPLIT = 1
                    x_plot = []
                    for j in range(self.mf_model.nfidelities):
                        x_plot.append((np.atleast_2d(x_test[i])))
                    X_plot = convert_x_list_to_array(x_plot)

                    hhf_mean_mf_model, hhf_var_mf_model = self.mf_model.model.predict(X_plot[(self.mf_model.nfidelities-1)*SPLIT:(self.mf_model.nfidelities)*SPLIT])
                    hhf_std_mf_model = np.sqrt(hhf_var_mf_model)

                    y_data.append(y_test[i])
                    x.append(i)
                    mfsm_model_mean=np.append(mfsm_model_mean,hhf_mean_mf_model[0,0])
                    mfsm_model_std=np.append(mfsm_model_std,hhf_std_mf_model[0,0])
                    if (y_test[i] < hhf_mean_mf_model[0][0]+hhf_std_mf_model[0][0]) and (y_test[i] > hhf_mean_mf_model[0][0]-hhf_std_mf_model[0][0]):
                            counter_1sigma += 1
                    if (y_test[i] < hhf_mean_mf_model[0][0]+2*hhf_std_mf_model[0][0]) and (y_test[i] > hhf_mean_mf_model[0][0]-2*hhf_std_mf_model[0][0]):
                            counter_2sigma += 1
                    if (y_test[i] < hhf_mean_mf_model[0][0]+3*hhf_std_mf_model[0][0]) and (y_test[i] > hhf_mean_mf_model[0][0]-3*hhf_std_mf_model[0][0]):
                            counter_3sigma += 1

            fig,ax = plt.subplots(figsize=(12, 2.5))
            #plt.bar(x=np.arange(len(mfsm_model_mean)), height=mfsm_model_mean, color="lightgray", label='RESuM')
            plt.fill_between(x=np.arange(len(mfsm_model_mean)), y1=mfsm_model_mean-3*mfsm_model_std, y2=mfsm_model_mean+3*mfsm_model_std, color="coral",alpha=0.2, label=r'$\pm 3\sigma$')
            plt.fill_between(x=np.arange(len(mfsm_model_mean)), y1=mfsm_model_mean-2*mfsm_model_std, y2=mfsm_model_mean+2*mfsm_model_std, color="yellow",alpha=0.2, label=r'$\pm 2\sigma$')
            plt.fill_between(x=np.arange(len(mfsm_model_mean)), y1=mfsm_model_mean-mfsm_model_std, y2=mfsm_model_mean+mfsm_model_std, color="green",alpha=0.2, label=r'RESuM $\pm 1\sigma$')
            plt.xlabel('HF Simulation Trial Number')
            ymin, ymax = ax.get_ylim()
            plt.ylim(ymin,ymax*1.3)
            plt.ylabel(r'$y_{raw}$')
            plt.plot(x[:],y_data[:],'.',color="black", label="HF Validation Data")
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [3,2,1,0]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=9, bbox_to_anchor=(0.665,1.),ncol=5)
            mse = mean_squared_error(y_data, mfsm_model_mean, squared=True)
            fig.text(0.905,0.5,f"within $1\sigma$: {counter_1sigma/len(y_data)*100.:.0f}% \nwithin $2\sigma$: {counter_2sigma/len(y_data)*100.:.0f}% \nwithin $1\sigma$: {counter_3sigma/len(y_data)*100.:.0f}% \nMSE {mse:.5f}")
            return fig, [counter_1sigma/len(y_data)*100.,counter_2sigma/len(y_data)*100.,counter_3sigma/len(y_data)*100.,mse]
        
    def draw_model(self):
            fig, ax = plt.subplots(self.mf_model.nfidelities, 1, figsize=(12, 10))
            for j,f in enumerate(self.mf_model.trainings_data):
                nsamples = 0
                mfsm_model_mean = np.empty(shape=[0, 0])
                mfsm_model_std = np.empty(shape=[0, 0])
                x_train = self.mf_model.trainings_data[f][0]
                y_train = self.mf_model.trainings_data[f][1]
                x_train, y_train = (np.atleast_2d(x_train), np.atleast_2d(y_train).T)
                y_data=[]
                x=[]
                for i in range(len(x_train)):
                        nsamples += 1
                        SPLIT = 1
                        
                        x_plot = (np.atleast_2d(x_train[i]))
                        X_plot = convert_x_list_to_array([x_plot , x_plot, x_plot])
                    
                        mean_mf_model, var_mf_model = self.mf_model.model.predict(X_plot[j*SPLIT:(j+1)*SPLIT])
                        std_mf_model = np.sqrt(var_mf_model)

                        y_data.append(y_train[i])
                        x.append(i)
                        mfsm_model_mean=np.append(mfsm_model_mean,mean_mf_model[0,0])
                        mfsm_model_std=np.append(mfsm_model_std,std_mf_model[0,0])

                ax[j].fill_between(x=np.arange(len(mfsm_model_mean)), y1=mfsm_model_mean-mfsm_model_std, y2=mfsm_model_mean+mfsm_model_std, color=self.colors_std[j],alpha=0.2, label=f'{f} model mean $\pm 1\sigma$')
                ax[j].plot(np.arange(len(mfsm_model_mean)),mfsm_model_mean,color=self.colors_mean[j], label=f'{f} model mean')
                ax[j].plot(x[:],y_data[:],'.',color="black", label="Data")
                ymin, ymax = ax[j].get_ylim()
                ax[j].set_ylim(ymin,ymax*1.1)
                handles, labels = ax[j].get_legend_handles_labels()
                
                order = [2,1,0]
                ax[j].legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=9, bbox_to_anchor=(0.665,1.),ncol=3)
                ax[j].set_ylabel('Data and Model Prediction')
                if j == (self.mf_model.nfidelities-1):
                    ax[j].set_xlabel('Data Point')
            return fig
    
