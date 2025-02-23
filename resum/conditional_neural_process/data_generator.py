import torch
import collections
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from collections import Counter
from tqdm import tqdm
import gc
from IPython.display import display, Image
import re
import sys
import yaml
import itertools
import h5py
from torch.utils.data import DataLoader, IterableDataset

from ..utilities import utilities as utils
import random

utils.set_random_seed(42)

class HDF5Dataset(IterableDataset):
    def __init__(self, hdf5_dir, 
                batch_size=3000, 
                files_per_batch=20,
                parameters = {'phi': {'key': "phi",'label_key': "phi_labels",'selected_labels': None},
                              'theta': {'key': "theta",'label_key': "theta_headers",'selected_labels': None},
                              'target': {'key': "target",'label_key': "target_labels",'selected_labels': None}}
        ):
        """
        - hdf5_dir: Directory containing HDF5 files.
        - batch_size: Number of samples per batch (3,400).
        - files_per_batch: Number of files used in each batch (34).
        """
        super().__init__()
        self.hdf5_dir = hdf5_dir
        self.batch_size = batch_size
        self.files_per_batch = files_per_batch
        self.rows_per_file = batch_size // files_per_batch
        self.epoch_counter = 0  # Tracks row block
        self.total_batches = 0
        self.parameters = parameters
        self.phi_selected_indices= None
        self.theta_selected_indices= None
        self.target_selected_indices= None

        # List and sort all HDF5 files
        self.files = sorted([os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith(".h5")])
        self.num_files = len(self.files)
        self.dataset_size =0 
        # Total row cycles per file to complete an epoch
        self.nrows = self.get_max_number_of_rows()
        self.total_cycles_per_epoch = self.nrows // self.rows_per_file  # nrows / k rows per batch = c cycles per full dataset pass
        
    def shuffle_files(self):
        """Shuffle the file order at the start of each full dataset pass (epoch)."""
        random.shuffle(self.files)
        self.epoch_counter = 0  # Reset row counter

    def get_max_number_of_rows(self):
        max_rows = 0
        self.dataset_size = 0 
        for file in self.files:
            with h5py.File(file, "r") as hdf:
                    if self.parameters['target']['key'] in hdf:
                        num_rows = hdf[self.parameters['target']['key']].shape[0]
                        self.dataset_size += num_rows
                        # Update max row count if this file has more rows
                        if num_rows > max_rows:
                            max_rows = num_rows
            if num_rows==0:
                print(f"WARNING! {file} has row size 0. Either no data or target key doesn't match.")
        if max_rows == 0:
                raise ValueError("ERROR! Data is either empty or target key doesn't match.")
        return max_rows
    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.dataset_size

    def __iter__(self):
        #print(f"Starting structured HDF5 loading for row block {self.epoch_counter}...")
        self.total_batches = 0
        batch_idx = 0
        used_rows = 0  # Track number of rows used

        while batch_idx < self.total_cycles_per_epoch:
            
            for i in range(0, len(self.files), self.files_per_batch):  # Loop over file chunks
                if i == 0  and self.epoch_counter==0:
                    self.phi_selected_indices=utils.read_selected_indices(self.files[0],self.parameters['phi'])
                    self.theta_selected_indices=utils.read_selected_indices(self.files[0],self.parameters['theta'])
                    self.target_selected_indices=utils.read_selected_indices(self.files[0],self.parameters['target'])

                batch = []
                selected_files = self.files[i:i + self.files_per_batch]
                # Select the next sequential k rows per file
                start_idx = self.epoch_counter * self.rows_per_file
                end_idx = start_idx + self.rows_per_file
                for file in selected_files:
                    with h5py.File(file, "r") as hdf:
                        phi=hdf[self.parameters['phi']['key']][start_idx:end_idx, self.phi_selected_indices]
                        if  len(hdf[self.parameters['theta']['key']][:].shape) == 1:
                            theta = hdf[self.parameters['theta']['key']][self.theta_selected_indices]
                            theta = np.tile(theta, (phi.shape[0], 1))
                        else:
                            theta = hdf[self.parameters['theta']['key']][start_idx:end_idx, self.theta_selected_indices]
                        features = np.hstack([theta, phi])
                        target=hdf[self.parameters['target']['key']][start_idx:end_idx, self.target_selected_indices]
                        
                        # Stack rows from this file
                        file_data = np.hstack([features, target])
                        batch.extend(file_data.tolist())

                        used_rows += len(file_data)

                # Yield batch of batch-size shuffled samples
                random.shuffle(batch)
                yield torch.tensor(batch, dtype=torch.float32)

                batch_idx += 1

            # Move to next row block
            self.epoch_counter += 1
            self.total_batches += batch_idx+1

            # If all files and rows (from k*i to k*(i+1)) are read, reshuffle files for the row block
            if self.epoch_counter >= self.total_cycles_per_epoch:
                print("Finished full dataset pass. Starting new epoch! ",self.epoch_counter)
                self.shuffle_files()
                break

CNPRegressionDescription = collections.namedtuple(
    "CNPRegressionDescription", ("query", "target_y")
)

class DataGeneration(object):
    """
    """
    def __init__(
        self,
        mode,
        config_file,
        path_to_files,
        batch_size,
        use_data_augmentation = False,
    ):
        self._context_ratio = config_file["cnp_settings"]["context_ratio"]
        self._batch_size = batch_size
        self.path_to_files = path_to_files
        self.dataloader="None"
        self.config_file=config_file

        _phi_key="phi"
        _theta_key="theta"
        _target_key="target"
        _names_theta=config_file["simulation_settings"]["theta_headers"]
        _names_phi=config_file["simulation_settings"]["phi_labels"]
        self._names_target =config_file["simulation_settings"]["target_label"]
        
        if not any(f.endswith(".h5") for f in os.listdir(path_to_files)):
            utils.convert_all_csv_to_hdf5(config_file)
        
        if mode != "config":
            if use_data_augmentation == "mixup":
                signal_condition = config_file["simulation_settings"]["signal_condition"]
                files = sorted([os.path.join(path_to_files, f) for f in os.listdir(path_to_files) if f.endswith(".h5")])
                print(f"Data Augmentation in Progress: Applying transformations...")
                for file in tqdm(files):
                    self.mixup_augment_data(file,config_file["cnp_settings"]["use_beta"],signal_condition)
                    _phi_key="phi_mixedup"
                    _target_key="target_mixedup"

        self.parameters={'phi': {'key': _phi_key,'label_key': "phi_labels",'selected_labels': _names_phi}, 
                        'theta': {'key': _theta_key,'label_key': "theta_headers",'selected_labels': _names_theta}, 
                        'target': {'key': _target_key,'label_key': "target_labels",'selected_labels': self._names_target}}

    def set_loader(self):
        dataset = HDF5Dataset(self.path_to_files, self._batch_size, files_per_batch=self.config_file["cnp_settings"]["files_per_batch"], parameters=self.parameters)
        self.dataloader = DataLoader(dataset, batch_size=None, num_workers=self.config_file["cnp_settings"]["number_of_walkers"], prefetch_factor=2) 

    def mixup_augment_data(self,filename, use_beta,condition_strings, seed=42):
        """
        Augments an imbalanced dataset using the "mixup" method for HDF5 files.

        Each background event is combined with a randomly drawn signal event using a weighted sum.
        The ratio is drawn from either a uniform distribution or a beta distribution.

        Args:
            filename (str): Path to the HDF5 file.
            use_beta (list or None): Distribution from which the ratio is drawn.
                - `None`: Uniform distribution in [0,1].
                - `[z1, z2]`: Beta distribution B(z1, z2).
            config_file (dict): Preloaded YAML config dictionary.

        Returns:
            None: Updates the existing HDF5 file with new datasets.
        """
        np.random.seed(seed)  # Set the seed for reproducibility
        with h5py.File(filename, "a") as f:  # Open in append mode
            # Check if mixup datasets already exist
            if "phi_mixedup" in f and "target_mixedup" in f:
                if "signal_condition" in f:
                    existing_conditions = [s.decode("utf-8") for s in f["signal_condition"][:]]
                    if existing_conditions == condition_strings:
                        return
            phi = np.array(f["phi"])  # Feature data
            target = np.array(f["target"])  # Labels
            has_weights = "weights" in f  # Check if "weights" dataset exists
            weights = np.array(f["weights"]) if has_weights else None

            # Identify background (0) and signal (1) indices
            background_indices = np.where(target == 0)[0]
            names_target_tmp=[label.decode("utf-8") if isinstance(label, bytes) else label for label in self._names_target]
            # Function to parse condition strings dynamically
            def parse_condition(condition_str, columns):
                """Parses condition strings and returns (column index, condition lambda)."""
                match = re.match(r"(\S+)\s*(==|!=|<=|>=|<|>)\s*(\S+)", condition_str)
                if not match:
                    raise ValueError(f"Invalid condition format: {condition_str}")

                column_name, operator, value = match.groups()
                if column_name not in columns:
                    raise ValueError(f"Column {column_name} not found in target!")

                column_idx = columns.index(column_name)  # Get the column index

                # Convert condition string to a lambda function
                return column_idx, lambda x: eval(f"x {operator} {value}", {"x": x})

            # Convert conditions to apply on NumPy target array
            conditions = np.ones(target.shape[0], dtype=bool)  # Start with all True

            for cond_str in condition_strings:
                col_idx, cond_func = parse_condition(cond_str, names_target_tmp)  # Get index and condition
                conditions &= cond_func(target[:, col_idx])  # Apply condition to the correct dimension

            # Find matching indices
            signal_indices = np.where(conditions)[0]
            # All indices in the dataset
            all_indices = np.arange(target.shape[0])

            # Background indices are those NOT in signal_indices
            background_indices = np.setdiff1d(all_indices, signal_indices)

            if len(background_indices) == 0 or len(signal_indices) == 0:
                raise ValueError("Dataset must contain both signal (1) and background (0) samples.")

            # Randomly pair each background sample with a signal sample
            sampled_signal_indices = np.random.choice(signal_indices, size=len(background_indices), replace=True)

            # Generate mixup ratios
            if use_beta and isinstance(use_beta, (list, tuple)) and len(use_beta) == 2:
                alpha = np.random.beta(use_beta[0], use_beta[1], size=(len(background_indices), 1))
            else:
                alpha = np.random.uniform(0, 1, size=(len(background_indices), 1))

            # Perform mixup augmentation
            phi_mixedup = alpha * phi[sampled_signal_indices] + (1 - alpha) * phi[background_indices]
            target_mixedup = alpha * target[sampled_signal_indices] + (1 - alpha) * target[background_indices]

            # Apply mixup to weights if they exist
            weights_mixedup = None
            if has_weights:
                weights_mixedup = alpha * weights[sampled_signal_indices] + (1 - alpha) * weights[background_indices]

            # Store new datasets in the same file
            if "phi_mixedup" in f:
                del f["phi_mixedup"]
            f.create_dataset("phi_mixedup", data=phi_mixedup, compression="gzip")

            if "target_mixedup" in f:
                del f["target_mixedup"]
            f.create_dataset("target_mixedup", data=target_mixedup, compression="gzip")

            if has_weights:
                if "weights_mixedup" in f:
                    del f["weights_mixedup"]
                f.create_dataset("weights_mixedup", data=weights_mixedup, compression="gzip")
            
            if "signal_condition" in f:
                del f["signal_condition"]
            f.create_dataset("signal_condition", data=np.array(condition_strings, dtype="S"))

    def format_batch_for_cnp(self,batch, context_is_subset=True):
        """
        Formats a batch into the query format required for CNP training with dynamic batch splitting.
        Parameters:
        - batch (torch.Tensor): Input batch of shape (batch_size, feature_dim).
        - total_batch_size (int): Expected full batch size (default: 3000).
        - context_ratio (float): Ratio of context points (default: 1/3).
        - target_ratio (float): Ratio of target points (default: 2/3).

        Returns:
        - CNPRegressionDescription(query=((batch_context_x, batch_context_y), batch_target_x), target_y=batch_target_y)
        """

        batch_size = batch.shape[0]  # Actual batch size (may be < 3000)
        
        # Dynamically compute num_context and num_target
        num_context = int(batch_size * self._context_ratio)
        num_target = batch_size - num_context  # Ensure it sums to batch_size

        # Shuffle the batch to ensure randomness
        batch = batch[torch.randperm(batch.shape[0])]

        # Split batch into input (X) and target (Y) features
        batch_x = batch[:, :-1]  # All features except last column (input features)
        batch_y = batch[:, -1]   # Last column is the target (output values)

        if context_is_subset:
            # **Context is taken as the first num_context points from target**
            batch_target_x = batch_x  # Target is the entire batch
            batch_target_y = batch_y  # Target outputs are the entire batch

            batch_context_x = batch_target_x[:num_context]  # Context is a subset of target
            batch_context_y = batch_target_y[:num_context]  # Context outputs
        else:
            # **Context and target are independent splits**
            batch_context_x = batch_x[:num_context]  # Context inputs
            batch_context_y = batch_y[:num_context]  # Context outputs
            batch_target_x = batch_x[num_context:num_context + num_target]  # Target inputs
            batch_target_y = batch_y[num_context:num_context + num_target]  # Target outputs

        # Ensure y tensors have correct dimensions (convert from 1D to 2D if needed)
        batch_context_y = batch_context_y.view(-1, 1) if batch_context_y.ndim == 1 else batch_context_y
        batch_target_y = batch_target_y.view(-1, 1) if batch_target_y.ndim == 1 else batch_target_y

        if batch_context_x.dim() == 2:  # Convert from [N, D] → [1, N, D]
            batch_context_x = batch_context_x.unsqueeze(0)
        if batch_context_y.dim() == 2:  # Convert from [N, 1] → [1, N, 1]
            batch_context_y = batch_context_y.unsqueeze(0)

        if batch_target_x.dim() == 2:  # Convert from [N, D] → [1, N, D]
            batch_target_x = batch_target_x.unsqueeze(0)
        if batch_target_y.dim() == 2:  # Convert from [N, 1] → [1, N, 1]
            batch_target_y = batch_target_y.unsqueeze(0)
        # Construct the query tuple
        query = ((batch_context_x, batch_context_y), batch_target_x)

        # Return the properly formatted object
        return CNPRegressionDescription(query=query, target_y=batch_target_y)

    def get_batch(self,batch_idx):
        """
        Retrieves a specific batch from an iterable DataLoader.

        Parameters:
        - dataloader (torch.utils.data.DataLoader): The DataLoader object.
        - batch_idx (int): The index of the batch to retrieve.

        Returns:
        - The requested batch.
        """
        batch = next(itertools.islice(self.dataloader, batch_idx, None))
        return self.format_batch_for_cnp(batch)

    def get_dataloader(self):
        return self.dataloader