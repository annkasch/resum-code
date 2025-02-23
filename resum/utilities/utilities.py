#!/usr/bin/env python3
import os
import sys
from termcolor import colored
import numpy as np
from tqdm import tqdm
import pandas as pd
import h5py
import random
import torch

def set_random_seed(seed=42):
    random.seed(seed)           # Python's built-in random module
    np.random.seed(seed)        # NumPy random seed
    torch.manual_seed(seed)     # PyTorch random seed

    # Ensures reproducibility on GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

def convert_single_csv_to_hdf5(csv_file, hdf5_file, theta_headers, target_label, weights_labels):
        """
        """
        # Read CSV file
        df = pd.read_csv(csv_file, index_col=0)

        # **Extract 'theta' data**
        
        theta_data = df[theta_headers].to_numpy()[0]
        weights_data = df[weights_labels].to_numpy()
        target_data = df[target_label].to_numpy()
        # **Extract 'phi' (all columns except theta_headers, 'target', 'weights')**
        phi_headers = [col for col in df.columns if col not in theta_headers + target_label + weights_labels + ['fidelity']]
        phi_data = df[phi_headers].to_numpy()

        # Filter out non-existing columns from weights
        existing_weights = [w for w in weights_labels if w in df.columns]
        # Select only available columns, or default to ones
        weights_data = df[existing_weights].to_numpy() if existing_weights else np.ones((len(df), 1))
        weights_labels = existing_weights if existing_weights else ["weights"]
        fidelity_data = df[['fidelity']].to_numpy() if 'fidelity' in df else np.ones((len(df), 1))

        # **Store Data in HDF5**
        with h5py.File(hdf5_file, "w") as hdf:
            hdf.create_dataset("fidelity", data=fidelity_data, compression="gzip")
            hdf.create_dataset("theta", data=theta_data, compression="gzip")
            hdf.create_dataset("theta_headers", data=np.array(theta_headers, dtype='S'), compression="gzip")
            hdf.create_dataset("phi", data=phi_data, compression="gzip")
            hdf.create_dataset("phi_labels", data=np.array(phi_headers, dtype='S'), compression="gzip")
            hdf.create_dataset("target", data=target_data, compression="gzip")
            hdf.create_dataset("target_labels", data=np.array(target_label, dtype='S'), compression="gzip")
            hdf.create_dataset("weights", data=weights_data, compression="gzip")
            hdf.create_dataset("weights_labels", data=np.array(weights_labels, dtype='S'), compression="gzip")

def convert_all_csv_to_hdf5(config_file):
        """
        Loops over all CSV files in `csv_dir` and converts each to an HDF5 file.
        
        Parameters:
        - csv_dir (str): Directory containing CSV files.
        - hdf5_dir (str): Directory where HDF5 files will be saved.
        - theta_headers (list of str): Column names that should be stored under 'theta'.
        
        Each CSV file is stored as an individual HDF5 file in `hdf5_dir`.
        """
        path_to_files=config_file["path_settings"]["path_to_test_files_lf"]

        theta_headers= config_file["simulation_settings"]["theta_headers"]
        target_label = config_file["simulation_settings"]["target_label"]
        weights_labels = config_file["simulation_settings"]["weights_labels"]

        # Loop through all CSV files
        print("Converting CSV to HDF5")
        for file in tqdm(sorted(os.listdir(path_to_files))):
            if file.endswith(".csv"):
                csv_file = os.path.join(path_to_files, file)
                hdf5_file = os.path.join(path_to_files, file.replace(".csv", ".h5"))  # Save as .hdf5
                # Call the function for single file conversion
                convert_single_csv_to_hdf5(csv_file, hdf5_file, theta_headers, target_label, weights_labels)

def read_selected_indices(hdf5_file, label_dict):
        """
        Gives back indices of only the specified columns from parameter_key inHDF5 file.
        label_dict is a dictionary with following structure {'key': "...",'label_key': "....",'selected_labels': ["radius","thickness",...]}
        """
        with h5py.File(hdf5_file, "r") as hdf:
            if label_dict['label_key'] not in hdf:
                selected_indices= [0] if len(hdf[label_dict['key']].shape) == 1 else list(range(hdf[label_dict['key']].shape[1]))
            else:
                selected_labels=label_dict['selected_labels']
                if selected_labels == None: 
                    selected_labels=hdf[label_dict['label_key']][:]

                labels = list(map(lambda x: x.decode() if isinstance(x, bytes) else x, hdf[label_dict['label_key']][:]))
                # Find indices of required columns **before** reading phi
                selected_indices = [labels.index(label) for label in selected_labels if label in labels]

            if not selected_indices:
                raise ValueError(f"None of the requested labels {selected_labels} exist in phi_labels!")
            return sorted(selected_indices)

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

        except FileNotFoundError:
            print(f"File not found: {f}")
        except Exception as e:
            print(f"Error processing file {f}: {e}")

    # Combine all rows into a single array
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