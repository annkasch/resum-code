import numpy as np
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

class AdaptiveSampling:
    def __init__(self, target_dist, initial_proposal, n_iterations, n_samples_per_iter, aggregated_dist=None, with_pca=True, condition=lambda x: True, with_KDE=True):
        """
        Initialize the AdaptiveSampling object.

        Parameters:
        - target_dist: function, computes the target distribution density f(phi | theta_k).
        - aggregated_dist: function, computes the aggregated proposal density g_agg(phi).
        - initial_proposal: list of dicts, parameters for the initial proposal distribution:
            [{'mean': ndarray, 'std': ndarray, 'weight': float}, ...].
        - n_iterations: int, number of adaptation iterations.
        - n_samples_per_iter: int, number of samples to draw per iteration.
        - condition: function, evaluates whether a sample satisfies the constraint.
        """
        self.target_dist = target_dist
        self.aggregated_dist = aggregated_dist
        self.with_aggre=True
        if self.aggregated_dist==None:
            self.with_aggre = False
        self.proposals = initial_proposal
        self.n_iterations = n_iterations
        self.n_samples_per_iter = n_samples_per_iter
        self.all_samples = []
        self.all_weights = []
        self.condition = condition
        self.with_KDE=with_KDE
        self.with_pca=with_pca
        self.entropy = []
        self.ess=[]
        self.weights_variance =[]
        self.kl_difergence = []

    def regularize_cov_matrix(self, cov_matrix, epsilon=1e-6):
        """
        Regularize a covariance matrix to ensure it is symmetric positive definite.
        """
        cov_matrix = np.atleast_2d(cov_matrix)  # Ensure 2D array
        return cov_matrix + np.eye(cov_matrix.shape[0]) * epsilon

    def sample_from_proposal(self):
        """Generate samples from the current proposal mixture with conditioned sampling."""
        samples = []
        for proposal in self.proposals:
            n_samples = max(1, int(proposal['weight'] * self.n_samples_per_iter))
            accepted_samples = []
            
            while len(accepted_samples) < n_samples:
                # Sanitize std values
                proposal['std'] = np.clip(np.nan_to_num(proposal['std'], nan=1e-6, posinf=1e6, neginf=1e-6), 1e-6, None)

                # Generate the covariance matrix
                cov_matrix = np.diag(proposal['std'] ** 2)

                # Regularize the covariance matrix if needed
                if not np.all(np.linalg.eigvals(cov_matrix) > 0):
                    cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])

                # Generate candidate samples
                candidate_samples = np.atleast_2d(multivariate_normal.rvs(
                    mean=proposal['mean'],
                    cov=cov_matrix,
                    size=n_samples,
                    random_state=42
                ))

                # Apply the condition
                filtered_samples = [sample for sample in candidate_samples if self.condition(sample)]
                accepted_samples.extend(filtered_samples[:n_samples - len(accepted_samples)])
            samples.extend(accepted_samples)
        return np.array(samples)

    def compute_log_proposal_density_with_pca(self,sample):
        log_proposal_density = None
        if self.with_pca==False:
            for p in self.proposals:
                # Compute the log PDF of this proposal component
                p['std'] = np.clip(p['std'], 1e-6, None)  # Avoid excessively small std
                cov_matrix = self.regularize_cov_matrix(np.diag(p['std'] ** 2))

                try:
                    component_log_pdf = multivariate_normal.logpdf(
                        sample,
                        mean=p['mean'],
                        cov=cov_matrix,
                        allow_singular=True
                    )
                    component_log_pdf += np.log(p['weight'])  # Include the component weight
                except ValueError as e:
                    print(f"PDF computation error: {e}")
                    component_log_pdf = -np.inf  # Log of zero density

                # Accumulate log-proposal density
                if log_proposal_density is None:
                    log_proposal_density = component_log_pdf
                else:
                    log_proposal_density = np.logaddexp(log_proposal_density, component_log_pdf)
        else:
            # Compute the proposal log-density using PCA-regularized covariance
            for p in self.proposals:
                # Regularize and sanitize covariance matrix
                p['std'] = np.clip(p['std'], 1e-6, None)  # Avoid excessively small std
                cov_matrix = self.regularize_cov_matrix(np.diag(p['std'] ** 2))

                # Perform PCA on the covariance matrix
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                threshold = 1e-6
                significant_indices = eigenvalues > threshold
                if not np.any(significant_indices):
                    print("Warning: No significant eigenvalues. Using full covariance.")
                    reduced_cov_matrix = cov_matrix  # Fallback to full covariance
                else:
                    reduced_eigenvectors = eigenvectors[:, significant_indices]
                    reduced_eigenvalues = eigenvalues[significant_indices]

                    # Reconstruct reduced covariance matrix
                    reduced_cov_matrix = self.regularize_cov_matrix(
                        reduced_eigenvectors @ np.diag(reduced_eigenvalues) @ reduced_eigenvectors.T
                    )

                # Transform the sample into PCA-reduced space
                transformed_sample = (
                    reduced_eigenvectors.T @ (sample - p['mean'])
                    if 'reduced_eigenvectors' in locals() else sample - p['mean']
                )

                # Compute the log PDF in the reduced space
                try:
                    proposal_log_pdf = multivariate_normal.logpdf(
                        transformed_sample,
                        mean=np.zeros(reduced_cov_matrix.shape[0]),
                        cov=reduced_cov_matrix,
                        allow_singular=True
                    )
                    proposal_log_pdf += np.log(p['weight'])  # Include mixture weight
                except ValueError as e:
                    print(f"PDF computation error: {e}")
                    proposal_log_pdf = -np.inf  # Log of zero density

                # Accumulate proposal log-density
                if log_proposal_density is None:
                    log_proposal_density = proposal_log_pdf
                else:
                    log_proposal_density = np.logaddexp(log_proposal_density, proposal_log_pdf)

        return log_proposal_density
    
    def compute_weights(self, samples, theta_k):
        """
        Compute importance weights for the given samples, accounting for the target,
        the proposal distribution, and the aggregated distribution.

        Parameters:
        - samples: ndarray, sampled phi values.
        - theta_k: ndarray, current design parameter.

        Returns:
        - weights: ndarray, adjusted importance weights.
        """
        log_weights = []

        for sample in samples:
            try:
                # Ensure sample is at least 1D
                sample = np.atleast_1d(sample)

                # Compute the target log-density
                target_density = np.squeeze(self.target_dist(sample, theta_k))  # Ensure scalar
                 
                target_density = max(target_density, 1e-12)
                log_target_density = np.log(target_density)

                log_proposal_density = self.compute_log_proposal_density_with_pca(sample)

                if self.with_aggre==True:
                    # Compute aggregated log-density
                    aggregated_density = np.squeeze(self.aggregated_dist(sample))
                    aggregated_density = max(aggregated_density, 1e-12)  # Avoid log(0)
                    log_aggregated_density = np.log(aggregated_density)

                    # Compute the combined log denominator (proposal + aggregated)
                    combined_log_density = np.logaddexp(log_proposal_density, log_aggregated_density)
                else:
                    combined_log_density = log_proposal_density

                # Compute importance weight in log space
                log_weight = log_target_density - combined_log_density
                log_weights.append(log_weight)

            except np.linalg.LinAlgError as e:
                print(f"Covariance matrix error: {e}")
                log_weights.append(-np.inf)  # Assign log zero weight for errors

        # Convert log weights to normalized weights
        log_weights = np.array(log_weights)
        max_log_weight = np.max(log_weights)  # For numerical stability during exponentiation
        stable_log_weights = log_weights - max_log_weight
        exp_weights = np.exp(stable_log_weights)

        if np.sum(exp_weights) == 0:
            raise ValueError("Sum of weights is zero, cannot normalize.")
        normalized_weights = exp_weights / np.sum(exp_weights)

        return normalized_weights
    
    def update_proposal(self, samples, weights):
        """Add a new proposal component based on weighted samples."""
        weights = np.nan_to_num(weights.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
        weights /= np.sum(weights)  # Normalize weights

        mean = np.average(samples, axis=0, weights=weights)
        std = np.sqrt(np.average((samples - mean) ** 2, axis=0, weights=weights))

        # Add the new proposal
        new_proposal = {'mean': mean, 'std': std*1.5, 'weight': 1.0 / (len(self.proposals) + 1)}
        self.proposals.append(new_proposal)

        # Normalize mixture weights
        for p in self.proposals:
            p['weight'] = 1.0 / len(self.proposals)

        # Shift the means of existing proposal components based on the weights
        #self.update_proposal_mean(samples,weights)

        # Shift the means of existing proposal components with kmeans
        #new_means = self.update_proposal_mean_with_weighted_kmeans(samples, weights, len(self.proposals))
        #for i, proposal in enumerate(self.proposals):
        #    proposal['mean'] = new_means[i]
    
    def update_proposal_mean_with_weighted_kmeans(self, samples, weights, n_clusters):
        weighted_kmeans = KMeans(n_clusters=n_clusters, n_init='auto',random_state=42)
        weighted_kmeans.fit(samples, sample_weight=weights)
        return weighted_kmeans.cluster_centers_
    
    def update_proposal_mean(self, samples, weights):
        idx_start =0
        for proposal in self.proposals:
            idx_end = idx_start+max(1, int(proposal['weight'] * self.n_samples_per_iter))
            proposal_samples = np.array(samples[idx_start:idx_end])  # Ensure samples are numpy arrays
            proposal_weights = np.array(weights[idx_start:idx_end])
            proposal_weights /= np.sum(proposal_weights)
            weighted_mean_shift = np.average(proposal_samples, axis=0, weights=proposal_weights)
            
            proposal['mean'] = (proposal['mean'] + weighted_mean_shift) / 2  # Weighted average with existing mean
            #proposal['std'] *= 1.1  # Slightly increase std to avoid overfitting
            idx_start = idx_end
    
    def update_proposal_with_kde(self, samples, weights, bandwidth=0.5):
        """
        Update proposal distribution using KDE on weighted samples.

        Parameters:
        - samples: ndarray, current samples.
        - weights: ndarray, importance weights for the samples.
        - bandwidth: float, bandwidth for the KDE.
        """
        # Normalize weights
        weights = np.array(weights).flatten()
        if np.sum(weights) == 0:
            raise ValueError("Sum of weights is zero, cannot normalize.")
        weights /= np.sum(weights)

        # Fit KDE to the weighted samples
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(samples, sample_weight=weights)

        # Sample new means from the KDE
        n_components = len(self.proposals)
        new_means = kde.sample(n_components, random_state=42)

        # Update proposals with new means and adjusted stds
        stds = np.sqrt(np.average((samples - np.mean(samples, axis=0))**2, axis=0, weights=weights)) + 1e-6
        for i, proposal in enumerate(self.proposals):
            proposal['mean'] = new_means[i]
            proposal['std'] = stds  # Same std for simplicity
            proposal['weight'] = 1.0 / n_components  # Equal weight

        # Optionally, add one new component to enhance exploration
        new_component = {
            'mean': np.average(samples, axis=0, weights=weights),
            'std': stds * 1.5,  # Slightly broadened std
            'weight': 1.0 / (n_components + 1)
        }
        self.proposals.append(new_component)

        # Normalize weights for all proposals
        for p in self.proposals:
            p['weight'] = 1.0 / len(self.proposals)

    def run(self, theta_k):
        """Execute the adaptive sampling process for a fixed theta_k."""
        for iteration in range(self.n_iterations):
            current_samples = self.sample_from_proposal()

            if self.with_KDE==True:
                weights = self.compute_weights(current_samples,theta_k)
                self.update_proposal(current_samples, weights)
                #self.update_proposal_with_kde(current_samples, weights, bandwidth=0.5)
            else: 
                weights = self.compute_weights(current_samples,theta_k)
                self.update_proposal(current_samples, weights)
                #self.update_proposal_with_kde(current_samples, weights, bandwidth=0.5)
                
                
            self.all_samples.append(current_samples)
            self.all_weights.append(weights)
            print(f"Iteration {iteration + 1}/{self.n_iterations}: New proposal added.")
            # Step 4: Track diagnostics
            self.ess.append(self.get_effective_sample_size(weights) / len(weights))
            self.entropy.append(self.get_entropy(weights))
            self.weights_variance.append(np.var(weights))
            print(f"ESS: {self.ess[-1]*100.:.2f} Entropy: {self.entropy[-1]:.2f}")

        self.proposals = self.proposals[:-1*(self.n_iterations-np.argmax(self.ess)-1)]
        # Normalize weights for all proposals
        for p in self.proposals:
            p['weight'] = 1.0 / len(self.proposals)
        self.n_iterations=np.argmax(self.ess)+1
        self.all_samples=self.all_samples[:self.n_iterations]
        self.all_weights=self.all_weights[:self.n_iterations]
        self.entropy=self.entropy[:self.n_iterations]
        self.ess=self.ess[:self.n_iterations]
        self.weights_variance[:self.n_iterations]
        return self.all_samples, self.all_weights, self.proposals
    
    def get_entropy(self, weights):
        return -np.sum(weights * np.log(weights + 1e-12))
    
    def get_effective_sample_size(self, weights):
        return 1 / np.sum(weights**2)

    #def get_kl_difergence(self,samples):
    #    return np.sum(self.target_dist(samples) * np.log(self.target_dist(samples) / self.proposals[-1]))

def estimate_aggregated_distribution(aggregated_phi, bandwidth='scott'):
    #kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(aggregated_dist)
    # Define a range of bandwidths to test
    if bandwidth=='GridSearchCV':
        bandwidths = np.logspace(-1, 1, 20)  # From 0.1 to 10

        # Use GridSearchCV to find the best bandwidth
        grid = GridSearchCV(
            KernelDensity(kernel='gaussian'),
            {'bandwidth': bandwidths},
            cv=5  # 5-fold cross-validation
        )
        grid.fit(aggregated_phi)
        bandwidth = grid.best_params_['bandwidth']
        print("Best bandwidth from GridSearchCV:", bandwidth)
    
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(aggregated_phi)
    def aggregated_dist(x):
        x = np.atleast_2d(x)
        return np.exp(kde.score_samples(x))
    return aggregated_dist

def initialize_multidimensional_proposal(data, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(data)
    proposals = []
    for i in range(n_clusters):
        cluster_data = data[kmeans.labels_ == i]
        mean = np.mean(cluster_data, axis=0)
        std = np.std(cluster_data, axis=0)
        weight = len(cluster_data) / len(data)
        proposals.append({'mean': mean, 'std': std, 'weight': weight})
    return proposals

def estimate_target_distribution(data, theta, bandwidth='scott'):
    if bandwidth=="GridSearchCV":
        bandwidths = np.logspace(-1, 1, 20)  # From 0.1 to 10
        # Use GridSearchCV to find the best bandwidth
        grid = GridSearchCV(
            KernelDensity(kernel='gaussian'),
            {'bandwidth': bandwidths},
            cv=5  # 5-fold cross-validation
        )
        grid.fit(data)
        bandwidth = grid.best_params_['bandwidth']
        print("Best bandwidth from GridSearchCV:", bandwidth)
    
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
    
    def target_distribution(x, theta):
        x = np.atleast_2d(x)
        return np.exp(kde.score_samples(x))
    return target_distribution

def initialize_proposals_with_kde(data, bandwidth=0.5, n_components=2):
    """
    Initialize Gaussian proposals based on KDE sampling.

    Parameters:
    - data: ndarray, shape (n_samples, n_features), data to fit KDE and sample from.
    - bandwidth: float, bandwidth for KDE.
    - n_components: int, number of Gaussian components to initialize.

    Returns:
    - proposals: list of dicts, each with 'mean', 'std', and 'weight' for a Gaussian proposal.
    """
    if bandwidth=='GridSearchCV':
        bandwidths = np.logspace(-1, 1, 20)  # From 0.1 to 10

        # Use GridSearchCV to find the best bandwidth
        grid = GridSearchCV(
            KernelDensity(kernel='gaussian'),
            {'bandwidth': bandwidths},
            cv=5  # 5-fold cross-validation
        )
        grid.fit(data)
        bandwidth = grid.best_params_['bandwidth']
        print("Best bandwidth from GridSearchCV:", bandwidth)

    # Fit KDE to the data
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)

    # Sample means from the KDE
    sampled_means = kde.sample(n_components, random_state=42)

    # Compute the covariance matrix for the KDE as an approximation of std
    stds = np.std(data, axis=0) + 1e-6  # Regularize to avoid zero std

    # Initialize Gaussian components with equal weights
    proposals = []
    for mean in sampled_means:
        proposals.append({'mean': mean, 'std': stds, 'weight': 1.0 / n_components})

    return proposals, kde