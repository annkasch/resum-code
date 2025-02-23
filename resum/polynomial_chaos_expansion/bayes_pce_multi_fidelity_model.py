import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
from itertools import combinations_with_replacement
from numpy.polynomial.legendre import Legendre
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from itertools import product, combinations
import random

# Set seeds for reproducibility
np.random.seed(42)         # NumPy seed
random.seed(42)            # Python random seed


class PCEMultiFidelityModel:
    def __init__(self, trainings_data, priors, degree=None):
        """
        Initialize the multi-fidelity model.

        Parameters:
        - basis_matrices (dict): Dictionary of basis matrices for each fidelity level.
          Example: {"lf": basis_matrix_lf, "mf": basis_matrix_mf, "hf": basis_matrix_hf}
        - trainings_data (dict): Dictionary of observed data for each fidelity level.
          Example: {"lf": [x_lf, y_lf], "mf": [x_mf,y_mf], "hf": [x_hf,y_hf]}
        - indices (dict): Dictionary of indices mapping one fidelity level to the next.
          Example: {"mf": indices_mf, "hf": indices_hf}
        - priors (dict): Dictionary of prior configurations for each fidelity level.
          Example: {"lf": {"sigma": 0.5}, "mf": {"sigma": 0.1}, "hf": {"sigma": 0.01}}
        """
        
        self.trainings_data = trainings_data
        self.fidelities = list(self.trainings_data.keys())
        self.priors = priors
        self.degree = degree
        if self.degree==None:
            self.find_optimal_order()
        self.model = None
        self.trace = None

        self.basis_matrices = {}
        self.indices = {}
        for f in self.fidelities:
            x_data = trainings_data[f][0]
            self.basis_matrices[f] = self._generate_basis(x_data)
        
        for i,f in enumerate(self.fidelities[1:]):
            self.indices[f] = self.find_indices(trainings_data[f][0],trainings_data[self.fidelities[i]][0])


    def _generate_basis(self, x_data):
        """
        Generate the multivariate Legendre basis for multi-dimensional inputs.

        Parameters:
        - x_data (ndarray): Input data of shape (n_samples, n_dim).

        Returns:
        - basis_matrix (ndarray): Shape (n_samples, n_terms).
        """
            
        n_samples, n_dim = x_data.shape
        terms = []

        # Generate all combinations of terms up to the given degree
        for deg in range(self.degree + 1):
            for combo in combinations_with_replacement(range(n_dim), deg):
                terms.append(combo)

        # Evaluate each term for all samples
        basis_matrix = np.zeros((n_samples, len(terms)))
        for i, term in enumerate(terms):
            poly = np.prod([Legendre.basis(1)(x_data[:, dim]) for dim in term], axis=0)
            basis_matrix[:, i] = poly

        return basis_matrix
    
    @staticmethod
    def find_indices(x_hf, x_lf):
        """
        Finds the indices of x_hf in x_lf.

        Parameters:
        - x_hf (numpy.ndarray): Array of high-fidelity x values.
        - x_lf (numpy.ndarray): Array of low-fidelity x values.

        Returns:
        - list: Indices of x_hf in x_lf.
        """
        indices = []
        for x in x_hf:
            idx = np.where((x_lf == x).all(axis=1))[0]
            if idx.size > 0:
                indices.append(idx[0])  # Append the index
            else:
                raise ValueError(f"Value {x} from x_hf not found in x_lf.")
        return indices
    
    @staticmethod
    def multivariate_legendre_with_interactions(order, x):
        """
        Generate multivariate Legendre polynomial basis with interaction terms.
        
        Parameters:
        - order (int): Maximum polynomial degree.
        - x (ndarray): Input data of shape (n_samples, n_features).
        
        Returns:
        - basis (ndarray): Basis matrix including interactions.
        """

        n_samples, n_features = x.shape
        degrees = list(product(range(order + 1), repeat=n_features))
        basis = []
        for degree in degrees:
            term = np.ones(n_samples)
            for i, d in enumerate(degree):
                term *= np.polynomial.legendre.Legendre.basis(d)(x[:, i])
            basis.append(term)

        # Add interaction terms
        for i, j in combinations(range(n_features), 2):
            basis.append(x[:, i] * x[:, j])

        return np.vstack(basis).T
    
    def _add_fidelity(self, model, fidelity, y_prev_pred_full):
        """
        Recursively add fidelity levels to the model.

        Parameters:
        - model (pm.Model): The PyMC model.
        - fidelity_chain (list): List of fidelities to be added (e.g., ["lf", "mf", "hf"]).
        - prev_pred (pm.Deterministic): The prediction from the previous fidelity level.

        Returns:
        - pm.Deterministic: Final prediction for the highest fidelity level.
        """

        # Basis matrix and observed data
        basis_matrix = self.basis_matrices[fidelity]
        observed = self.trainings_data[fidelity][1]

        y_prev_pred_subset = pm.Deterministic(f"y_prev_pred_subset_{fidelity}", y_prev_pred_full[self.indices[fidelity]])
        # Scaling factor
        rho = pm.Normal(f"rho_{fidelity}", mu=1, sigma=self.priors[fidelity]["sigma_rho"])
        # Priors for high-fidelity discrepancy coefficients
        coeffs_delta = pm.Normal(f"coeffs_delta_{fidelity}", mu=0, sigma=self.priors[fidelity]["sigma_coeffs_delta"], shape=self.basis_matrices[fidelity].shape[1])
        # High-fidelity discrepancy
        delta_pred = pm.Deterministic(f"delta_{fidelity}", pm.math.dot(self.basis_matrices[fidelity], coeffs_delta))
            
        # High-fidelity predictions
        y_pred = pm.Deterministic(f"y_pred_{fidelity}", rho * y_prev_pred_subset + delta_pred)
        # Likelihood for high-fidelity data
        sigma = pm.HalfNormal(f"sigma_{fidelity}", sigma=self.priors[fidelity]["sigma"])
        y_like = pm.Normal(f"y_like_{fidelity}", mu=y_pred, sigma=sigma, observed=self.trainings_data[fidelity][1])
        return y_pred

    def build_model(self):
        """
        Build the PyMC multi-fidelity model recursively.
        """
          # ["lf", "mf", "hf"]
        with pm.Model() as model:
            # Start with low-fidelity coefficients

            coeffs = pm.Normal(f"coeffs_{self.fidelities[0]}",
                mu=0,
                sigma=self.priors[self.fidelities[0]]["sigma_coeffs"],
                shape=self.basis_matrices[self.fidelities[0]].shape[1]
            )
            y_prev_pred_full = pm.Deterministic(f"y_pred_full_{self.fidelities[0]}", pm.math.dot(self.basis_matrices[self.fidelities[0]], coeffs))
            sigma = pm.HalfNormal(f"sigma_{self.fidelities[0]}", sigma=self.priors[self.fidelities[0]]["sigma"])
            y_like = pm.Normal(f"y_like_{self.fidelities[0]}", mu=y_prev_pred_full, sigma=sigma, observed=self.trainings_data[self.fidelities[0]][1])
            
            # Add fidelities recursively
            for fidelity in self.fidelities[1:]:
                y_prev_pred_full = self._add_fidelity(model, fidelity, y_prev_pred_full)
            self.model = model
    
    def run_inference(self, method="advi", n_samples=2000, n_steps=1000000):
        """
        Run inference on the PCE model.

        Parameters:
        - method (str): Inference method ("advi" or "nuts").

        Returns:
        - pm.backends.base.MultiTrace: The posterior samples.
        """
        if self.model is None:
            raise RuntimeError("Model has not been built. Call build_model() first.")

        with self.model:
            if method == "advi":
                # Variational Inference
                approx = pm.fit(n=n_steps, method="advi", progressbar=True)
                self.trace = approx.sample(n_samples)
            elif method == "nuts":
                # HMC Sampling
                self.trace = pm.sample(self.n_samples, tune=1000, target_accept=0.95, cores=4)
            else:
                raise ValueError(f"Unknown inference method: {method}")

        return self.trace
    
    def plot_trace(self):
        az.plot_trace(self.trace)
        plt.show()
    
    def evaluate_mse(self, x_test, y_test):
        '''
        - test_data: Dictionary of training data for fidelity level to test.
          Example: [x_hf,y_hf]
        '''
        if self.trace is None:
            raise RuntimeError("Model has not been trained. Call run_inference() first.")

        # Generate the basis matrix for the test data

        basis_matrix_test = self._generate_basis(x_test)
        # Posterior mean predictions
        coeffs_mean = self.trace.posterior[f"coeffs_{self.fidelities[0]}"].mean(dim=["chain", "draw"]).values
        y_pred = np.dot(basis_matrix_test, coeffs_mean)
        
        for fidelity in self.fidelities[1:]:
            coeffs_delta_mean = self.trace.posterior[f"coeffs_delta_{fidelity}"].mean(dim=["chain", "draw"]).values
            rho_mean = self.trace.posterior[f"rho_{fidelity}"].mean(dim=["chain", "draw"]).values
            # Predict y_test
            y_pred = rho_mean * y_pred + np.dot(basis_matrix_test, coeffs_delta_mean)

        mse = np.mean((y_pred - y_test) ** 2)
        return mse

    def plot_validation(self, x_data, y_true):
        """
        Plot the validation data with the uncertainty prediction bands.

        Parameters:
        - x_data (ndarray): Input data (e.g., validation or test set).
        - y_true (ndarray): True high-fidelity target values for validation.
        - trace: Trace object containing posterior samples from PyMC.
        """
        y_hf_pred_samples = self.generate_y_hf_pred_samples(x_data)
        hf_sample_numbers = np.arange(len(y_true))

        plt.figure(figsize=(10, 6))
        plt.fill_between(
            hf_sample_numbers,
            np.percentile(y_hf_pred_samples, 0.5, axis=0),
            np.percentile(y_hf_pred_samples, 99.5, axis=0),
            color="coral", alpha=0.2, label=r'$\pm 3\sigma$'
        )
        plt.fill_between(
            hf_sample_numbers,
            np.percentile(y_hf_pred_samples, 2.5, axis=0),
            np.percentile(y_hf_pred_samples, 97.5, axis=0),
            color="yellow", alpha=0.2, label=r'$\pm 2\sigma$'
        )
        plt.fill_between(
            hf_sample_numbers,
            np.percentile(y_hf_pred_samples, 16, axis=0),
            np.percentile(y_hf_pred_samples, 84, axis=0),
            color="green", alpha=0.2, label=r'$\pm 1\sigma$'
        )
        plt.scatter(hf_sample_numbers, y_true, marker='.', color="black", label="HF Validation Data")

        plt.xlabel("HF Simulation Trial Number")
        plt.ylabel(r"$y_{hf}$")
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [3, 2, 1, 0]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
        plt.title("Validation Data with Prediction Uncertainty Bands")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def validate_coverage(self, x_data, y_true):
        """
        Validate the coverage of the model for 1, 2, and 3 sigma intervals.

        Parameters:
        - y_true (ndarray): True high-fidelity target values for validation.
        - y_hf_pred_samples (ndarray): Posterior predictive samples for high-fidelity predictions.

        Returns:
        - dict: Percentages of validation data within 1, 2, and 3 sigma intervals.
        """
        y_hf_pred_samples = self.generate_y_hf_pred_samples(x_data)
        counters = {1: 0, 2: 0, 3: 0}

        # Calculate percentile intervals for the posterior samples
        percentiles = {
            1: (np.percentile(y_hf_pred_samples, 16, axis=0), np.percentile(y_hf_pred_samples, 84, axis=0)),
            2: (np.percentile(y_hf_pred_samples, 2.5, axis=0), np.percentile(y_hf_pred_samples, 97.5, axis=0)),
            3: (np.percentile(y_hf_pred_samples, 0.5, axis=0), np.percentile(y_hf_pred_samples, 99.5, axis=0)),
        }

        # Count the number of y_true points within each interval
        for i, y in enumerate(y_true):
            for sigma in [1, 2, 3]:
                low, high = percentiles[sigma]
                if low[i] <= y <= high[i]:
                    counters[sigma] += 1

        # Calculate percentages
        coverage = {sigma: (counters[sigma] / len(y_true)) * 100 for sigma in [1, 2, 3]}
        return coverage
    

    def generate_y_hf_pred_samples(self, x_data):
        """
        Generate high-fidelity prediction samples based on posterior trace.

        Parameters:
        - x_data (ndarray): Input data (e.g., validation or test set).
        - trace: Trace object containing posterior samples from PyMC.

        Returns:
        - y_hf_pred_samples (ndarray): Predicted high-fidelity samples (shape: n_samples_total x n_hf_samples).
        """
        basis_matrix_test = self._generate_basis(x_data)  # Shape: (n_samples, n_terms_hf)

        coeff_samples = self.trace.posterior[f"coeffs_{self.fidelities[0]}"].values  # Shape: (n_chains, n_draws, n_terms_hf)
        coeff_samples_flat = coeff_samples.reshape(-1, coeff_samples.shape[-1])  # Shape: (n_samples_total, n_terms_hf)
        y_pred_samples = np.dot(coeff_samples_flat, basis_matrix_test.T)  # Shape: (n_samples_total, n_lf_samples)

        for fidelity in self.fidelities[1:]:

            # Extract coefficients from the posterior
            coeff_samples_delta = self.trace.posterior[f"coeffs_delta_{fidelity}"].values  # Shape: (n_chains, n_draws, n_terms_hf)
            coeff_samples_delta_flat = coeff_samples_delta.reshape(-1, coeff_samples_delta.shape[-1])  # Shape: (n_samples_total, n_terms_hf)
            delta_pred_samples = np.dot(coeff_samples_delta_flat, basis_matrix_test.T)  # Shape: (n_samples_total, n_hf_samples)
            rho_samples = self.trace.posterior[f"rho_{fidelity}"].values  # Shape: (n_chains, n_draws)
            rho_samples_flat = rho_samples.flatten()  # Shape: (n_samples_total,)
            
            # Compute HF predictions
            y_pred_samples = rho_samples_flat[:, None] * y_pred_samples + delta_pred_samples  # Shape: (n_samples_total, n_hf_samples)

        return y_pred_samples
    
    def find_optimal_order(self, max_order=5, n_splits=5):
        """
        Find the optimal polynomial order using cross-validation.

        Parameters:
        - max_order (int): Maximum polynomial order to test.
        - n_splits (int): Number of splits for cross-validation.

        Returns:
        - optimal_order (int): Optimal polynomial order.
        """
        print("Finding the optimal polynomial order using cross-validation...")
        errors = []
        kf = KFold(n_splits=n_splits)

        for order in range(1, max_order + 1):
            # Generate basis for LF and HF
            basis_with_interactions = {}
            c = {}
            for fidelity in self.fidelities:
                basis_with_interactions[fidelity]=self.multivariate_legendre_with_interactions(order, self.trainings_data[fidelity][0])
                c[fidelity] = np.linalg.lstsq(basis_with_interactions[fidelity],self.trainings_data[fidelity][1], rcond=None)[0]
                # Predict LF contributions to HF
            y_pred = {}
            for i,fidelity in enumerate(self.fidelities[1:]):
                y_pred[f"{self.fidelities[i]}_{fidelity}"] = basis_with_interactions[fidelity] @ c[self.fidelities[i]]
                delta = self.trainings_data[fidelity][1] - y_pred[f"{self.fidelities[i]}_{fidelity}"]

                mse_fold = []
                x_train = {}; y_train = {}
                x_test  = {}; y_test = {}

                for train_idx, test_idx in kf.split(self.trainings_data[fidelity][0]):
                    x_train[fidelity], x_test[fidelity] = self.trainings_data[fidelity][0][train_idx], self.trainings_data[fidelity][0][test_idx]
                    y_train[fidelity], y_test[fidelity] = self.trainings_data[fidelity][1][train_idx], self.trainings_data[fidelity][1][test_idx]

                    # Generate basis matrices for train and test
                    phi_train = self.multivariate_legendre_with_interactions(order, x_train[fidelity])
                    phi_test = self.multivariate_legendre_with_interactions(order, x_test[fidelity])

                    # Fit HF correction
                    c_hf = np.linalg.lstsq(phi_train, y_train[fidelity]-phi_train @ c[self.fidelities[i-1]], rcond=None)[0]
                    y_pred_fold = phi_test @ c_hf + phi_test @ c[self.fidelities[i-1]]

                    mse_fold.append(mean_squared_error(y_test[fidelity], y_pred_fold))

                # Append mean error for this order
                errors.append(np.mean(mse_fold))

        # Return the optimal order (1-based indexing for order)
        optimal_order = np.argmin(errors) + 1
        print(f"The optimal order is {optimal_order}")
        return optimal_order