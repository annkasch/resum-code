#torch==2.1.2
#numpy==1.24.3
#re==2.2.1
import torch
import collections
import matplotlib.pyplot as plt
import numpy as np
import dask.dataframe as dd
import os
from imblearn.over_sampling import SMOTE
from collections import Counter
from tqdm import tqdm
import gc
from IPython.display import display, Image
import re
import sys
sys.path.append('../utilities')
import utilities as utils


def readin_data(filelist, usecols=[]):
    """This function reads in all data from a filelist and gives back a dataframe

    Args:
        filelist: a list of filenames (*.csv)
        usecols: a subset of column numbers to read-in

    Returns:
        df: dataframe of the data of the given filelist
    """

    if len(filelist) == 0:
        raise SystemError("Empty filelist. No neutron tier files found. Run tier file generation first.")

    it =0
    print("Reading in data files...")
    for file in tqdm(filelist):
        ### read in data

        if usecols != []:
            df_tmp = dd.read_csv(file, delimiter=',', usecols=usecols)
        else:
            df_tmp = dd.read_csv(file, delimiter=',')
        if it == 0:
            df = df_tmp
        else:
            df = dd.concat([df, df_tmp], axis=0, ignore_index=True)
        it+=1
    return df

        
def split_signal_bkg(filename, name_y, split_values=[1,0]):
    """This function splits the tabular data in a file into signal and background when column "name_y" equals the split values

    Args:
        filename: name of file containing tabular data
        name_y: name of colume which contains signal/ bkg labels
        split_values: contains the split values: split when signal is equal to value[0]; bkg is equal to value[1]
    """

    bkg_df = dd.read_csv(f"{filename}", delimiter=',',index_col = False)
    #bkg_df.sample(frac=1).reset_index(drop=True)

    signal_df = bkg_df.loc[bkg_df[name_y] == split_values[0]]
    signal_df = signal_df.reset_index(drop=True)

    
    bkg_df = bkg_df.loc[bkg_df[name_y] == split_values[1]]
    bkg_df = bkg_df.reset_index(drop=True)


    #print("Writing split data to csv ... ")
    signal_df.to_csv(f"{filename}_signal")
    bkg_df.to_csv(f"{filename}_bkg")

    del [[bkg_df,signal_df]]
    gc.collect()
    bkg_df=dd.from_array(np.array([], dtype=float))
    signal_df=dd.from_array(np.array([], dtype=float))
    
    #print("...done")


def generate_data(path_to_files, ratio_testing, fout, no_shuffle=False):
    """ This function reads in all data from a filelist, shuffles the data randomly, 
    splits it into training and testing set and writes two separat files containing
    the datasets.

    Args:
        path_to_files: path to *csv data files
        ratio_testing: spliting ratio between the testing and training datasets
        fout: basename of the output files
    """
    #print(sys.getrecursionlimit())
    sys.setrecursionlimit(10000)
    
    filelist = utils.get_all_files(path_to_files)
    df = readin_data(filelist=filelist)

    x_tmp = df.to_dask_array().compute()

    if no_shuffle == False:
        print("Shuffling data...")
        np.random.shuffle(x_tmp)

    if ratio_testing > 0:
        npoints = int(df.shape[0].compute())
        n_training = int(np.floor(npoints * (1-ratio_testing)))

        df_testing = dd.from_array(x_tmp[n_training:],columns=df.columns)
        df_testing.to_parquet(f"{fout[0:fout.rindex('_')]}_testing")
        x_tmp = x_tmp[0:n_training]

        # free memory
        del [df_testing]
        gc.collect()
        df_testing=dd.from_array(np.array([], dtype=float))

    df = dd.from_array(x_tmp,columns=df.columns)
    df.to_parquet(f"{fout[0:fout.rindex('_')]}_training")
    del [df]
    gc.collect()

    df = dd.from_array(np.array([], dtype=float))
 
    
def smote_augment_data(filename, names_x, name_y, sig_bkg_ratio, plotting = True):
    """This function boost the minority class in the imbalanced dataset using SMOTE. 
    This minority oversampling method will create synthetic samples of the minority class 
    by using the feature space to generate new instances with the help of interpolation 
    between the positive instances that lie together fitting. The function writes a 
    transformed version of the dataset to file.

    Args:
        filename:
        names_x: The column names containing x. The k-nearest neighbors of x are obtained 
        by calculating the Euclidean distance between x and every other sample in the set
        name_y: The column name which contains the classifications of x
        sign_bkg_ratio: the boosting ratio of the minority class

    Returns:
        transformed version of our dataset
    """

    df = dd.read_csv(f"{filename}/*", delimiter=',')

    y = (df[name_y]).compute()
    counts = Counter(y)
    # the ratio of events with more than 1 secondary capture is < 1e-5 and will be neglected
    y = np.select([y > 1], [1],y)
    X = df[names_x].compute()

    npoints = len(df)
    num, den = sig_bkg_ratio[1:].split('to')
    r_sig_bkg = float(num)/float(den)
    nnew = int(npoints*r_sig_bkg)

    X_1 = X[0:nnew]
    y_1 = y[0:nnew]
    X_2 = X[nnew:-1]
    y_2 = y[nnew:-1]

    # transform the dataset
    print("Augmenting training using SMOTE...")
    oversample = SMOTE()
    X_1, y_1 = oversample.fit_resample(X_1, y_1)
    
    X = np.append(X_1,X_2,axis=0)
    y = np.append(y_1,y_2,axis=0)

    if plotting == True:
        counts_new = Counter(y)
        ax1 = plt.subplot(221)
        ax1.bar(counts.keys(), counts.values())
        ax1.set_title(f'{counts}', fontsize=6)

        ax2 = plt.subplot(222)
        ax2.bar(counts_new.keys(), counts_new.values())
        ax2.set_title(f'{counts_new}', fontsize=6)
        plt.savefig(f"{filename}_smote{sig_bkg_ratio}.png")
        plt.show() 
        plt.close()

    df_new = dd.from_array(X, columns=names_x)
    #df_new.insert(len(names_x),name_y, y, True)
    y = dd.from_array(y)  
    df_new[name_y] = y
    # shuffle dataframe

    arr_tmp = df_new.to_dask_array().compute()
    np.random.shuffle(arr_tmp)
    np.random.shuffle(arr_tmp)
    np.random.shuffle(arr_tmp)
    df_new = dd.from_array(arr_tmp, columns = df_new.columns)

    #print("Writing training data to csv ...")

    df_new.to_csv(f"{filename}_smote{sig_bkg_ratio}")

    del [df_new]
    gc.collect()
    df_new=dd.from_array(np.array([], dtype=float))

def mixup_augment_data(filename, use_beta):
    """This function improves the imbalanced dataset according to the "mixup" method. The function loops 
    over a list of files. For each background event in a file it builds a linear combination with a randomly 
    drawn signal event in the same file. The ratio between the background and signal event is a number 
    between 0 and 1, drawn either from a uniform distribution or from a beta function. The new data is
    written to file.

    Args:
        path_to_files: path to files
        use_beta: defines the distribution from which the ratio is drawn 
                    * use_beta = None => ratio uniformly distributed in [0,1]
                    * use_beta = [z1,z2] => ratio is drawn from a beta function B(z1,z2)

    Returns: 
    """
    df_bkg = dd.read_csv(f"{filename}", delimiter=',', index_col = False)
    df_sig = df_bkg.loc[df_bkg["nC_Ge77"] == 1]
    df_bkg = df_bkg.loc[df_bkg["nC_Ge77"] == 0]
    del df_sig["Unnamed: 0"]
    del df_bkg["Unnamed: 0"]

    names=df_bkg.columns
    x_bkg = df_bkg[names].to_dask_array().compute()
    x_sig = df_sig[names].to_dask_array().compute()

    indices = np.random.randint(0,len(x_sig),len(x_bkg))
    x_sig=np.array([x_sig[i] for i in indices])

    if use_beta != None and len(use_beta)==2:
        alpha = np.random.beta(use_beta[0],use_beta[1], size=(len(df_bkg),1))
    else:
        print("Error: use_beta has incorrect dimension")
    
    x = alpha * x_sig + (1-alpha) * x_bkg
    df_new = dd.from_array(x, columns=names)        

    fout = filename.replace("/tier2/", f"/tier3/beta_{use_beta[0]}_{use_beta[1]}/")
    fout = fout.replace("tier2", "tier3")

    df_new=df_new.repartition(npartitions=1)
    df_new.to_csv([fout])
    del [[df_bkg,df_sig,df_new]]
    gc.collect()
    df_bkg=dd.from_array(np.array([], dtype=float))
    df_sig=dd.from_array(np.array([], dtype=float))
    df_new = dd.from_array(np.array([], dtype=float))

CNPRegressionDescription = collections.namedtuple(
    "CNPRegressionDescription", ("query", "target_y")
)

class DataGeneration(object):
    """
    """
    def __init__(
        self,
        num_iterations, 
        num_context_points, 
        num_target_points,
        batch_size,
        x_size,
        y_size,
        path_to_files,
        names_x,
        name_y,
        mode,
        use_data_augmentation= None,
        config_wise = False,
        ratio_testing = 1/10,
        sig_bkg_ratio = "_1to2",
        filename = f'CNP_data'
    ):
        self._num_iterations = num_iterations
        self._num_context_points = num_context_points
        self._num_target_points = num_target_points
        self._batch_size = batch_size
        self._x_size = x_size
        self._y_size = y_size
        self.use_data_augmentation = use_data_augmentation
        self._sig_bkg_ratio = sig_bkg_ratio
        if self.use_data_augmentation == "mixup":
             self._sig_bkg_ratio = 0
        self._names_x = names_x
        self._name_y = name_y
        self._ratio_testing = ratio_testing # define the splitting between training and testing data
        path_tmp = path_to_files[0:path_to_files.rindex("/")]
        self._filename = f"{path_tmp}/.data_management/{filename}_{mode}"
        self._num_total_points = 50000
        self._partition_size = 40
        
        if mode != "config":
            os.system(f'mkdir -p {path_tmp}/.data_management')
            if os.path.exists(self._filename) == False:
                if self.use_data_augmentation == "mixup" and mode == "training":
                    self._ratio_testing = 0.
                generate_data(path_to_files+"neutron", self._ratio_testing, self._filename, config_wise)
            
            if mode == "training":
                if self.use_data_augmentation == "smote":
                    filename_tmp = f'{self._filename}_{self.use_data_augmentation}{sig_bkg_ratio}'
                    # read in all files and augment data for a balanced ratio between signal and background    
                    if os.path.exists(f'{filename_tmp}') == False:
                        smote_augment_data(self._filename, self._names_x, self._name_y, self._sig_bkg_ratio)
                    elif os.path.exists(f'{path_tmp}/.data_management/{filename_tmp}.png') == True:
                        display(Image(filename=f'{path_tmp}/.data_management/{filename_tmp}.png'))
                    self._filename = filename_tmp

            self.generate_partitions(self._filename)
            self._filelist = utils.get_all_files(f'{self._filename}/',ending=".parquet")
            
            #self._num_total_points = 0
            #for file in self._filelist:
            #    with open(file, "rbU") as f:
            #        self._num_total_points += int(np.floor(sum(1 for _ in f)/ self._batch_size))

            # get number of total points in file
        else:
            self._partition_size = 1
            self._filelist = utils.get_all_files(path_to_files,ending=".csv")

        
    def generate_partitions(self, path_to_files):
        min_num_files = int(np.floor(self._num_iterations/self._partition_size))
        npoints = (self._num_context_points + self._num_target_points)* self._partition_size
        self._filename = f"{path_to_files}_{self._num_target_points+self._num_context_points}_tmp"

        if os.path.exists(self._filename) == True:
            nfiles = len(utils.get_all_files(f'{self._filename}/',".parquet"))
            with open(f"{self._filename}/part.{(nfiles-1):04d}.parquet", "rbU") as f:
                npoints_last = int(sum(1 for _ in f))-1

            if nfiles > 0 and nfiles < min_num_files:
                self.add_partitions(path_to_files, npoints, nfiles, min_num_files)
            
            elif nfiles >= min_num_files:

                max_iter = int((nfiles*self._partition_size) + np.floor(npoints_last/(npoints)))
                if self._num_iterations > max_iter:
                    print(f"Warning: Iteration number is larger then maximal iterations available in dataset.")
                    self._num_iterations = max_iter
                    print(f"Setting number of iterations to {self._num_iterations}")
                    min_num_files = int(np.floor(self._num_iterations/self._partition_size))
                         
        if os.path.exists(self._filename) == False:
            self.add_partitions(path_to_files, npoints, 0)
    
    def add_partitions(self, path_to_files, npoints, start, end=-1):

        names = self._names_x.copy()
        if isinstance(self._name_y,str):
            names.append(self._name_y)
        else:
            for name in self._name_y:
                names.append(name)

        df = dd.read_parquet(f"{path_to_files}/*", usecols=names)
        #df = dd.read_csv(f"{path_to_files}/*", delimiter=',', usecols=names)
        if set(self._names_x).issubset(df.columns) == False:
            self.clean_up()
            self.generate_partitions(path_to_files)
            return
        print("Repartitioning data...")
        df = df.repartition(npartitions=1).reset_index(drop=True)
        x_arr = df.to_dask_array().compute()

        total_points = len(x_arr)
        num_files = int(np.ceil(total_points/npoints))

        if end > num_files or end==-1:
            end = num_files
        max_iter = int(np.floor(df.shape[0].compute()/(self._num_target_points + self._num_context_points)))
        if self._num_iterations > max_iter:
                print(f"Warning: Iteration number is larger then maximal iterations available in dataset.")
                self._num_iterations = max_iter
                print(f"Setting number of iterations to {self._num_iterations}")

        npoints = (self._num_context_points + self._num_target_points)*self._partition_size
        x_arr = df.to_dask_array().compute()
        
        for it in tqdm(range(start,end)):
                x = x_arr[0:npoints]
                if (len(x) % npoints != 0) or len(x)==0:
                    nrows = int(np.floor(len(x)/(self._num_target_points+self._num_context_points)))
                    if nrows == 0:
                        continue
                    x = x_arr[0:nrows*(self._num_target_points+self._num_context_points)]

                df_new = dd.from_array(x, columns=df.columns)
                df_new = df_new.repartition(npartitions=1)
                #dd.to_parquet(df_new, f"{self._filename}/part.{it}.parquet")
                name_function = lambda x: f"part.{int(x+it):04d}.parquet"
                df_new.to_parquet({self._filename}, name_function=name_function)
                x_arr = x_arr[npoints:]
    
    def get_data(self, iteration, context_is_subset=False):
        num_target_points_tmp = self._num_target_points 
        if context_is_subset == True:
            num_target_points_tmp += self._num_context_points
        batch_context_x = np.array([[np.zeros(self._x_size) for i in range(self._num_context_points)]])
        batch_context_y = np.array([[np.zeros(self._y_size) for i in range(self._num_context_points)]])
        batch_target_x = np.array([[np.zeros(self._x_size) for i in range(num_target_points_tmp)]])
        batch_target_y = np.array([[np.zeros(self._y_size) for i in range(num_target_points_tmp)]])

        for batch in range(self._batch_size):
            # read in partition containing iteration data
            fname = self._filelist[int(np.floor(iteration/self._partition_size))]
            if (fname.split('.')[-1]=="parquet"):
                df = dd.read_parquet(fname,ignore_metadata_file=True)
            else:
                df = dd.read_csv(fname)
            
            # Select the targets
            row_start = int((iteration%self._partition_size)*(self._num_target_points+self._num_context_points))
            arr_df_x = df[self._names_x].to_dask_array().compute()
            arr_df_y = df[self._name_y].to_dask_array().compute()
            # the ratio of events with more than 1 secondary capture is < 1e-5 and will be neglected
            arr_df_y = np.select([arr_df_y > 1], [1],arr_df_y)
            target_y = arr_df_y[row_start:row_start+num_target_points_tmp]
            
            if isinstance(self._name_y,str):
                target_y = np.array( [ [float(x)] for x in target_y])
            target_x = arr_df_x[row_start:row_start+num_target_points_tmp]

            batch_target_x  = np.append(batch_target_x,[target_x],axis=0)
            batch_target_y  = np.append(batch_target_y,[target_y],axis=0)

            if (self._num_context_points > 0):
                # Select the observations
                row_start = row_start+self._num_target_points
                context_y = arr_df_y[row_start:row_start+self._num_context_points]
                if isinstance(self._name_y,str):
                    context_y = np.array([ [float(x)] for x in context_y])
                context_x = arr_df_x[row_start:row_start+self._num_context_points]

                batch_context_y = np.append(batch_context_y,[context_y],axis=0)
                batch_context_x = np.append(batch_context_x,[context_x],axis=0)
                
            # free memory from dataframes
            del [df]
            gc.collect()
            df=dd.from_array(np.array([], dtype=float))

        batch_context_y = torch.from_numpy(batch_context_y[1:]).float()
        batch_context_x = torch.from_numpy(batch_context_x[1:]).float()
        batch_target_x = torch.from_numpy(batch_target_x[1:]).float()
        batch_target_y = torch.from_numpy(batch_target_y[1:]).float()
        
        query = ((batch_context_x, batch_context_y), batch_target_x)
    
        if self._num_context_points == 0 :
            query = query[1][0]
            batch_target_y = batch_target_y[0]

        return CNPRegressionDescription(query=query, target_y=batch_target_y)
    
    def build_tier3(self, path_to_tier2, use_beta = None):
        filelist = utils.get_all_files(path_to_tier2+"neutron")
        for file in tqdm(filelist):

            if self.use_data_augmentation == "smote":
                fout = path_to_tier2.replace("/tier2/", f"/tier3/smote{self._sig_bkg_ratio}/")
                os.system(f'mkdir -p {fout}')
                smote_augment_data(file, self._names_x, self._name_y, self._sig_bkg_ratio, plotting=False)
                fout=file.replace("/tier2/", f"/tier3/smote{self._sig_bkg_ratio}/")
                fout=fout.replace("tier2","tier3")
                frename = f'{file}_smote{self._sig_bkg_ratio}'
                os.system(f'mv {frename} {fout}')

            elif self.use_data_augmentation == "mixup":
                fout = path_to_tier2.replace("/tier2/", f"/tier3/beta_{use_beta[0]}_{use_beta[1]}/")
                os.system(f'mkdir -p {fout}')
                mixup_augment_data(file, use_beta)

    def clean_up(self):
        os.system(f'rm -r {self._filename}*_tmp')