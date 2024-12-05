import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from scipy.optimize import minimize, NonlinearConstraint
import GPy
from emukit.multi_fidelity import kernels
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import (
    convert_x_list_to_array,
    convert_xy_lists_to_arrays
)
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.acquisition import Acquisition
from emukit.experimental_design.acquisitions import IntegratedVarianceReduction, ModelVariance
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.loop.candidate_point_calculators import SequentialPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer

# Ensure reproducibility
np.random.seed(123)


class MFGPModel():
    def __init__(self, trainings_data, noise):

        self.trainings_data = trainings_data
        self.fidelities = list(self.trainings_data.keys())
        self.noise = noise
        self.model = None

    def build_model(self,n_restarts=100):
        """
        Constructs and trains a linear multi-fidelity model using Gaussian processes.
        """
        x_train_tmp = []
        y_train_tmp = []
        for fidelity in self.fidelities:
            x_train_tmp.append(self.trainings_data[fidelity][0])
            y_train_tmp.append(self.trainings_data[fidelity][1])
        
        X_train, Y_train = convert_xy_lists_to_arrays(x_train_tmp, y_train_tmp)

        # Define kernels for each fidelity
        kernels_list = [
            GPy.kern.RBF(X_train[0].shape[0] - 1),
            GPy.kern.RBF(1),
            GPy.kern.RBF(X_train[0].shape[0] - 1),
            GPy.kern.RBF(1)
        ]

        lin_mf_kernel = kernels.LinearMultiFidelityKernel(kernels_list)
        gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=len(self.fidelities))

        # Fix noise terms for each fidelity
        for i,fidelity in enumerate(self.fidelities):
            # Construct the attribute name dynamically
            noise_attr = f"Gaussian_noise" if i == 0 else f"Gaussian_noise_{i}"
            try:
                # Access and fix the noise attribute
                getattr(self.gpy_lin_mf_model.mixed_noise, noise_attr).fix(self.noise[fidelity])
            except AttributeError:
                print(f"Error: Attribute '{noise_attr}' not found in the model.")
                raise

        # Wrap and optimize the model
        self.model = GPyMultiOutputWrapper(
            gpy_lin_mf_model, len(self.fidelities), n_optimization_restarts=n_restarts, verbose_optimization=True
        )
        self.model.optimize()

    def set_data(trainings_data_new):

        X_train, Y_train = convert_xy_lists_to_arrays([x_train_l,x_train_m, x_train_h], [y_train_l,y_train_m, y_train_h])
        self.model.set_data(X_train, Y_train)




# --- Custom Acquisitions ---
class Cost(Acquisition):
    def __init__(self, costs):
        self.costs = costs

    def evaluate(self, x):
        fidelity_index = x[:, -1].astype(int)
        return np.array([self.costs[i] for i in fidelity_index])[:, None]

    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x.shape)


class InequalityConstraints(Acquisition):
    def __init__(self):
        pass

    def evaluate(self, x):
        delta_x = np.ones(len(x))
        for i, xi in enumerate(x[:, :-1]):
            if plotting.get_inner_radius(xi) < 90.0:
                delta_x[i] = 0.0
            elif plotting.get_outer_radius(xi) > 265.0:
                delta_x[i] = 0.0
            elif plotting.get_outer_radius(xi) - plotting.get_inner_radius(xi) > 20.0:
                delta_x[i] = 0.0
            elif (
                xi[2] * xi[1] * xi[4]
                > 1.05 * np.pi * (plotting.get_outer_radius(xi)**2 - plotting.get_inner_radius(xi)**2)
            ):
                delta_x[i] = 0.0
            elif plotting.is_crossed(xi):
                delta_x[i] = 0.0
            else:
                delta_x[i] = 1.0
        return delta_x[:, None]

    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x[:, :-1].shape)


# --- Utility Functions ---
def add_sample(x_train, y_train, mode, labels, ylabel, sample, version="v1"):
    """
    Adds a sample to the training dataset.
    """
    data = pd.read_csv(f"in/Ge77_rates_new_samples_{version}.csv")
    rows = list(
        set(data[data["Sample"] == sample].index)
        .intersection(data[data["Mode"] == mode].index)
    )
    fidelity = "HF" if mode == 1.0 else "LF"
    for i in rows:
        x_new = [data.iloc[i][l] for l in labels]
        x_train = np.append(x_train, [x_new], axis=0)
        y_train = np.append(y_train, [[data.iloc[i][ylabel]]], axis=0)
        print(f"Adding {fidelity} sample at {x_new} with {ylabel} of {y_train[-1]}")
    return x_train, y_train
