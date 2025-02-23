import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import sys
import os
from . import plotting_utils as plotting

def plot(prediction_y_training, target_y_training, loss_training, prediction_y_testing, target_y_testing,  loss_testing, it=None):
        
        index_list_1 = np.where(target_y_testing > 0.5)[0]
        target_signal_testing = np.array([target_y_testing[i] for i in index_list_1])
        prediction_signal_testing = np.array([prediction_y_testing[i] for i in index_list_1])
        target_bkg_testing = np.delete(target_y_testing, index_list_1)
        prediction_bkg_testing = np.delete(prediction_y_testing, index_list_1)

        #######################
        index_list_2 = np.where(target_y_training > 0.5)[0]
        target_signal_training = np.array([target_y_training[i] for i in index_list_2])
        prediction_signal_training = np.array([prediction_y_training[i] for i in index_list_2])
        target_bkg_training = np.delete(target_y_training, index_list_2)
        prediction_bkg_training = np.delete(prediction_y_training, index_list_2)

        fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
        if it != None:
                fig.suptitle(f'Training Iteration {it}', fontsize=10)
        # plot testing 
        bins = 100 
        if len(target_signal_testing)>0:
                ax[1].hist(target_signal_testing, range=[0.0, 1.0],bins=bins, color='orangered', alpha=1.0, label='label (signal)')
        ax[1].hist(target_bkg_testing, range=[0.0, 1.0],bins=bins, color=(3/255,37/255,46/255), alpha=0.8, label='label (bkg)')
        ax[1].hist(prediction_bkg_testing, range=[0.0, 1.0],bins=bins, color=(113/255,150/255,159/255), alpha=0.8, label='network (bkg)')
        if len(target_signal_training)>0:
                ax[1].hist(prediction_signal_testing, range=[0.0, 1.0],bins=bins,  color='coral', alpha=0.8,label='network (signal)')    
       

        ax[1].set_yscale('log')
        ax[1].set_ylabel("Count")
        ax[1].set_xlabel(r'$y_{CNP}$')
        ax[1].set_title(f'Testing (loss {loss_testing})', fontsize=10)
        #ax[1].legend(loc='bottom left', bbox_to_anchor=(0, -1.5))
        #ax[1].text(0.35,-0.25, f'Test data {Counter([i[0] for i in target_y_testing])}', fontsize=8.5, transform=ax[1].transAxes)

        #plot training
        if len(target_signal_training)>0:
                ax[0].hist(target_signal_training, range=[0.0, 1.0],bins=bins, color='orangered', alpha=1.0, label='label (signal)')
        ax[0].hist(target_bkg_training, range=[0.0, 1.0],bins=bins, color=(3/255,37/255,46/255), alpha=0.8, label='label (bkg)')
        ax[0].hist(prediction_bkg_training, range=[0.0, 1.0],bins=bins, color=(113/255,150/255,159/255), alpha=0.8, label='network (bkg)')
        if len(target_signal_training)>0:
                ax[0].hist(prediction_signal_training, range=[0.0, 1.0],bins=bins, color='coral', alpha=0.8, label='network (signal)')
        
        ax[0].set_yscale('log')
        ax[0].set_ylabel("Count")
        ax[0].set_xlabel(r'$y_{CNP}$')
        ax[0].set_title(f'Training (loss {loss_training})', fontsize=10)

        fig.subplots_adjust(bottom=0.3, wspace=0.33)
        if len(target_signal_training)>0:
                ax[1].legend(labels=['label (signal)', 'label (bkg)', 'network (signal)', 'network (bkg)'],loc='upper center', 
                bbox_to_anchor=(-0.15, -0.1),fancybox=False, shadow=False, ncol=2)
        else:
                ax[1].legend(labels=['label', 'network'],loc='upper center', 
                bbox_to_anchor=(-0.15, -0.1),fancybox=False, shadow=False, ncol=2)
        plt.show()

        return fig


def plot_config(prediction_y, target_y, loss, param=""):
    index_list = np.where(target_y == 1)[0]
    target_signal = np.array([target_y[i] for i in index_list])
    prediction_signal = np.array([prediction_y[i] for i in index_list])
    target_bkg = np.delete(target_y, index_list)
    prediction_bkg = np.delete(prediction_y, index_list)
    
    #fig, ax = plt.subplots(nrows=2, figsize=(6,6))
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4,1]})
    
    ax[0].hist(target_bkg, range=[0.0, 1.0],bins=100, color=(3/255,37/255,46/255), alpha=0.8, label='label (bkg)')
    ax[0].hist(prediction_bkg, range=[0.0, 1.0],bins=100, color=(113/255,150/255,159/255), alpha=0.8, label='network (bkg)')
    ax[0].hist(target_signal, range=[0.0, 1.0],bins=100, color='orangered', alpha=1.0, label='label (signal)')
    ax[0].hist(prediction_signal, range=[0.0, 1.0],bins=100, color='coral', alpha=0.8, label='network (signal)')
    
    
    ax[0].set_title(f'Testing (loss {loss}) {param}{Counter([i[0] for i in target_y])}', fontsize=10)
    ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Count")
    ax[0].set_xlabel(r"$y_{CNP}$")
    ax[1].set_axis_off()

    return fig



def sum_hist(prediction_y, target_y, htsig, htbkg, hpsig, hpbkg):

    index_list = np.where(target_y > 0.5)[0]
    target_signal = np.array([target_y[i] for i in index_list])
    prediction_signal = np.array([prediction_y[i] for i in index_list])
    target_bkg = np.delete(target_y, index_list)
    prediction_bkg = np.delete(prediction_y, index_list)

    bins = len(htsig)
    range = [0.0,1.0]

    h = np.histogram(target_signal, range=range,bins=bins)[0]
    htsig = htsig + h
    h = np.histogram(target_bkg, range=range, bins=bins)[0]
    htbkg = htbkg + h
    h = np.histogram(prediction_signal, range=range, bins=bins)[0]
    hpsig = htsig + h
    h += np.histogram(prediction_bkg, range=range, bins=bins)[0]
    hpbkg = hpbkg + h

    return htsig, htbkg, hpsig, hpbkg


def plot_result_summed(htsig, htbkg, hpsig, hpbkg):
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4,1]})
    nbins = len(htsig)
    range = [0.0, 1.0]
    bin_length = (range[1]-range[0])/nbins
    bins = np.arange(range[0],range[1]+bin_length, bin_length)
    centroids = (bins[1:] + bins[:-1]) / 2

    ax[0].hist(centroids, weights = htbkg, range=range, bins=nbins, color=(3/255,37/255,46/255), alpha=0.8, label='label (bkg)')
    ax[0].hist(centroids, weights = hpbkg, range=range, bins=nbins, color=(113/255,150/255,159/255), alpha=0.8, label='network (bkg)')
    ax[0].hist(centroids, weights = htsig, range=range, bins=nbins, color='orangered', alpha=1.0, label='label (signal)')
    ax[0].hist(centroids, weights = hpsig, range=range, bins=nbins, color='coral', alpha=0.8, label='network (signal)')
    
    ax[0].set_title(f'Testing', fontsize=10)
    ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Count")
    ax[0].set_xlabel(r"$y_{CNP}$")


    ax[1].set_axis_off()

    return fig

def get_subplot_result_configwise(ax, prediction_y, target_y, loss, param=""):
    index_list = np.where(target_y == 1)[0]

    target_signal = np.array([target_y[i] for i in index_list])
    prediction_signal = np.array([prediction_y[i] for i in index_list])
    target_bkg = np.delete(target_y, index_list)
    prediction_bkg = np.delete(prediction_y, index_list)
    
    ax.hist(target_bkg, range=[0.0, 1.0],bins=100, color=(3/255,37/255,46/255), alpha=0.8, label='label (bkg)')
    ax.hist(prediction_bkg, range=[0.0, 1.0],bins=100, color=(113/255,150/255,159/255), alpha=0.8, label='network (bkg)')
    ax.hist(target_signal, range=[0.0, 1.0],bins=100, color='orangered', alpha=1.0, label='label (signal)')
    ax.hist(prediction_signal, range=[0.0, 1.0],bins=100, color='coral', alpha=0.8, label='network (signal)')
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)
    ax.set_yscale("log")
    ax.set_ylabel("Count")
    ax.set_xlabel(r"$y_{CNP}$")



def plot_result_configwise(prediction_y, target_y, loss, x):
    sum_sim=np.sum(target_y)
    mean_pred=np.mean(prediction_y)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    fig.suptitle(f'Testing (loss {loss}) {Counter([i[0] for i in target_y])} mean(CNP): {mean_pred:.3f}, sum(Sim): {sum_sim:.0f}', fontsize=12)
    get_subplot_result_configwise(ax[0],prediction_y, target_y, loss)

    plotting.get_subplot_moderator(ax[1],x)
    tmp_str=f"r={x[0]}, d={x[1]}, n={x[2]}, " + r"$\varphi$=" + f"{x[3]}, L={x[4]}"
    ax[1].legend([tmp_str],loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)
        
    return fig