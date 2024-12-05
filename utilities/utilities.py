#!/usr/bin/env python3
import os
import sys
import argparse
from termcolor import colored
import numpy as np
from tqdm import tqdm
import pandas as pd


def get_all_files(path_to_files, ending='.csv'):
    """This function finds all file in a directory with a given ending

    Args:
        path_to_files: files path
        ending: define ending of the files

    Returns:
        res: a list with all filenames
    """

    dir_path = './'

    filename = ""
    index=[i+1 for i in range(len(path_to_files)) if  path_to_files[i]=='/']

    if len(index)>0:
        dir_path=path_to_files[:index[-1]]
        filename=path_to_files[index[-1]:]

    res = []

    filelist=os.listdir(dir_path)
    filelist.sort()
    #filelist = sorted(filelist, key=int)
    for file in filelist:

        if file.startswith(filename) and file.endswith(ending):
            res.append(f'{dir_path}{file}')
    if len(res) == 0:
        print(f"Warning: No files found at {path_to_files}.")
    return res

def get_dataframes_concat(path, nrows=None):
    file_list = get_all_files(path,".csv")
    df_new = pd.DataFrame()
    for file_in in tqdm(file_list):
        if nrows==None:
            df = pd.read_csv(file_in, index_col=0)
        else:
            df = pd.read_csv(file_in, nrows=nrows, index_col=0)
        # Append to the main DataFrame
        x_labels=["x_0[m]","y_0[m]","z_0[m]","px_0[m]","py_0[m]","pz_0[m]","ekin_0[eV]"]
        y_label = 'nC_Ge77'
        df_sig = df[(df[y_label]>=1)][x_labels]
        print(len(df_sig))
        df_new = pd.concat([df_new, df], ignore_index=True)
    return df_new, len(file_list)

def get_all_signal_events(filename_base, nrows):
    # Set parameter name/x_labels -> needs to be consistent with data input file
    x_labels=["x_0[m]","y_0[m]","z_0[m]","px_0[m]","py_0[m]","pz_0[m]","ekin_0[eV]"]
    y_label = 'nC_Ge77'

    x_lf_list_sig=[]
    file_list = get_all_files(filename_base)
    for f in tqdm(range(len(file_list))):
        print(f)
        try:
            # Read the file
            data_train = pd.read_csv(f,nrows=nrows)
            
            # Find rows where y_label == 1
            row_lf_sig = data_train.index[data_train[y_label] == 1]
            # Extract corresponding rows and append to the list
            x_lf_list_sig.append(data_train.loc[row_lf_sig][x_labels].to_numpy())
            # Find rows where y_label == 1
            #row_lf = data_train.index[data_train[y_label] < 5]
            # Extract corresponding rows and append to the list
            #x_lf_list.append(data_train.loc[row_lf][x_labels].to_numpy())
            #y_lf_list.append(data_train.loc[row_lf][y_label].to_numpy())

        except FileNotFoundError:
            print(f"File not found: {f}")
        except Exception as e:
            print(f"Error processing file {f}: {e}")

    # Combine all rows into a single array
    #x_lf_all = np.vstack(x_lf_list)
    #y_lf_all = np.vstack(y_lf_list)
    x_lf_sig_all = np.vstack(x_lf_list_sig)

    # Output the result
    print(f"Total rows with y(x) = 1: {x_lf_sig_all.shape[0]}")

def INFO(output):
    try:
        print(colored('[INFO] '+output, 'green'))
    except:
        print(colored('[INFO] '+str(output), 'green'))

def WARN(output):
    try:
        print(colored('[WARNING] '+output, 'yellow'))
    except:
        print(colored('[WARNING] '+str(output), 'yellow'))

def ERROR(output):
    try:
        print(colored('[ERROR] '+output, 'red'))
    except:
        print(colored('[ERROR] '+str(output), 'red'))
    sys.exit()

