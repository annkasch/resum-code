import numpy as np
import os
import pandas as pd
import sys
sys.path.append('/global/cfs/projectdirs/legend/users/aschuetz/analysis/legend-multi-fidelity-surrogate-model/utilities')
import simulation_utils as sim
import utilities as utils
from pathlib import Path
#import adaptive_sampling_aggregated as ais
np.random.seed(42)
random_seed = 42
from tqdm import tqdm
import adaptive_sampling as ais
import argparse

def condition(x):
    #print(x[0:3],np.sqrt(np.power(x[0],2)+np.power(x[1],2)))
    if np.sqrt(np.power(x[0],2)+np.power(x[1],2)) > 2.8:
        return 0
    if x[2] > 2. or x[2]<-4.:
        return 0
    if x[6] < 0:
        return 0
    return 1


def main(start, end, v_out, v_in, n_samples, iter, n_rows, path_in):
    
    version_out = v_out
    version_in = v_in
    nrows=n_rows
    bandwidth=0.5
    n_clusters=2
    with_KDE=True
    n_samples_per_iter=n_samples
    n_iterations = 10
    ais_sample_iteration = iter
    path_to_repo = "/global/cfs/projectdirs/legend/users/aschuetz/analysis/legend-multi-fidelity-surrogate-model/adaptive_sampling"
    Path('{}/out/LF/{}'.format(path_to_repo,version_out)).mkdir(parents=True, exist_ok=True)
    Path('{}/out/LF/{}/out'.format(path_to_repo,version_out)).mkdir(parents=True, exist_ok=True)

    for j in range(start,end):
        print("Running {}".format(j))
        # Open a file for writing
        n_LF = j
        
        path_to_files = f'{path_in}/neutron-sim-LF-{version_in}-{n_LF:04}-tier2_'
        data_train, sample_iter = utils.get_dataframes_concat(path_to_files, df_new=pd.DataFrame(), niterations=ais_sample_iteration, nrows=nrows)
        sys.stdout = open(f'{path_to_repo}/out/LF/{version_out}/out/neutron-ais-output_{n_LF:04d}_{version_out}.txt', "a")
        print("****************************************************************************************************")
        print(f"Starting Adaptive Importance Sampling Trial {n_LF:04d} Iteration {sample_iter} KDE {with_KDE} Refinement Iterations {n_iterations}")
        print("****************************************************************************************************")
        
        # Set parameter name/x_labels -> needs to be consistent with data input file
        x_labels=["x_0[m]","y_0[m]","z_0[m]","px_0[m]","py_0[m]","pz_0[m]","ekin_0[eV]"]
        y_label = 'nC_Ge77'

        x_lf = data_train[x_labels].to_numpy()
        x_lf_sig = data_train[(data_train[y_label] >= 1)][x_labels].to_numpy()

        # Output the result
        print(f"Total rows with y(x) = 1: {x_lf_sig.shape[0]}/ {x_lf.shape[0]}")
        theta=data_train[["radius","thickness","npanels","theta","length"]].to_numpy()[0]
        target_distribution = ais.estimate_target_distribution(x_lf,theta, bandwidth='scott')
        
        if with_KDE==True:
            aggreated_dist=None
            proposals, kde = ais.initialize_proposals_with_kde(x_lf_sig, bandwidth=bandwidth, n_components=n_clusters)
        else: 
            data_train_all, sample_iter = utils.get_dataframes_concat(path_to_files[:-11], df_new=pd.DataFrame(), niterations=None,nrows=nrows, ending=f"-tier2_0000.csv")
            for k in range(1,ais_sample_iteration):
                data_train_all, sample_iter = utils.get_dataframes_concat(path_to_files[:-11], df_new=data_train_all, niterations=None,nrows=nrows, ending=f"-tier2_{k:04d}.csv")
            x_lf_sig_all = data_train_all[(data_train_all[y_label] >= 1)][x_labels].to_numpy()

            print(f"Total rows with y(x) = 1: {x_lf_sig_all.shape[0]}")
            aggreated_dist = ais.estimate_aggregated_distribution(x_lf_sig_all, bandwidth='scott')
            # Initialize Gaussian proposals from 5D data
            proposals = ais.initialize_multidimensional_proposal(x_lf_sig_all, n_clusters=n_clusters)
            print(f"Total rows with y(x) = 1: {x_lf_sig_all.shape[0]}")
            #initial_samples = kde.sample(n_samples, random_state=42)
            for proposal in proposals:
                proposal['std'] *= 1.5  # Increase standard deviation

        # Output the proposals
        print("Gaussian Proposals:")
        for i, proposal in enumerate(proposals):
            print(f"Cluster {i + 1}:")
            print(f"  Mean: {proposal['mean']}")
            print(f"  Std:  {proposal['std']}")
            print(f"  Weight: {proposal['weight']:.2f}")

        # Initialize the AdaptiveSampling class
        adaptive_sampler = ais.AdaptiveSampling(
            target_dist=target_distribution,
            initial_proposal= proposals.copy(),
            aggregated_dist=aggreated_dist,
            n_iterations=n_iterations,
            n_samples_per_iter=int(n_samples_per_iter*1.01),
            condition=condition,
            with_pca=True,
            with_KDE=with_KDE
        )
        # Run the adaptive sampling process
        samples, weights, proposals = adaptive_sampler.run(theta_k=theta)
        print("Final Proposals:")
        for i, proposal in enumerate(proposals):
            mean = ", ".join([f"{v:.2f}" for v in proposal['mean']])
            std = ", ".join([f"{v:.2f}" for v in proposal['std']])
            print(f"Component {i + 1}: Mean=[{mean}], Std=[{std}], Weight={proposal['weight']:.4f}")

        # Value to extend
        samples_new = samples[-1][0:n_samples_per_iter]
        samples_tmp= [np.append(row,0.0) for row in samples_new]
        x=np.array(samples_tmp)
        
        weights_new = weights[-1][0:n_samples_per_iter]
        weights_new /= np.sum(weights_new)  

        df_out = pd.DataFrame({
        'x[m]': ["{:.5f}".format(val) for val in x[:, 0]],
        'y[m]': ["{:.5f}".format(val) for val in x[:, 1]],
        'z[m]': ["{:.5f}".format(val) for val in x[:, 2]],
        'xmom[m]': ["{:.5f}".format(val) for val in x[:, 3]],
        'ymom[m]': ["{:.5f}".format(val) for val in x[:, 4]],
        'zmom[m]': ["{:.5f}".format(val) for val in x[:, 5]],
        'ekin[eV]': ["{:.5f}".format(val) for val in x[:, 6]],
        'time[ms]': ["{:.5f}".format(val) for val in x[:, 7]]
        })
        df_out['weights']=["{:.5e}".format(val) for val in weights_new]
        df_out.to_csv(f'/global/cfs/projectdirs/legend/users/aschuetz/simulation/data/neutron-inputs-{version_out}/neutron-inputs-design0_{len(x)}_{n_LF:04d}_{sample_iter:04d}_{version_out}.dat', sep = " ",header=True)
        sim.print_geant4_macro_adaptive(theta, n_LF, f'/global/cfs/projectdirs/legend/users/aschuetz/simulation/macros/Neutron-Simulation-LF-{version_out}',mode='LF', version=version_out)
        
        print("Weights min:", np.min(weights_new))
        print("Weights max:", np.max(weights_new))
        print("Weights mean:", np.mean(weights_new))
        print("Weights std:", np.std(weights_new))
        print("****************************************************************************************************")
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--n_rows', type=int, default=None)
    parser.add_argument('--version_in', type=str, default='v1.4')
    parser.add_argument('--version_out', type=str, default='v1.7')
    parser.add_argument('--path_to_files', type=str, default='./')
    parser.add_argument('--iteration', type=int, default=1)
    args = parser.parse_args()
    main(args.start, args.end, args.version_out, args.version_in, args.n_samples, args.iteration, args.n_rows,args.path_to_files)