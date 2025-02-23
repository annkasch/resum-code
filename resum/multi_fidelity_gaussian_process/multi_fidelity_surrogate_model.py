import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
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
from emukit.core.loop.candidate_point_calculators import SequentialPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
import copy

# Ensure reproducibility
np.random.seed(123)


class MFGPModel():
    def __init__(self, trainings_data, noise, inequality_constraints=None):
        self.trainings_data = copy.deepcopy(trainings_data)
        self.fidelities = list(self.trainings_data.keys())
        self.nfidelities = len(self.fidelities)
        self.noise = noise
        self.model = None
        if inequality_constraints==None:
            self.inequality_constraints=InequalityConstraints()
        else:
            self.inequality_constraints=inequality_constraints

    def set_traings_data(self, trainings_data):
        self.trainings_data = copy.deepcopy(trainings_data)

    def build_model(self,n_restarts=10):
        """
        Constructs and trains a linear multi-fidelity model using Gaussian processes.
        """
        x_train = []
        y_train = []
        for fidelity in self.fidelities:
            x_tmp=np.atleast_2d(self.trainings_data[fidelity][0])
            y_tmp=np.atleast_2d(self.trainings_data[fidelity][1]).T
            x_train.append(x_tmp)
            y_train.append(y_tmp)
        
        X_train, Y_train = convert_xy_lists_to_arrays(x_train, y_train)

        # Define kernels for each fidelity
        kernels_list = []
        for f in range(self.nfidelities-1):
            kernels_list.append(GPy.kern.RBF(X_train[0].shape[0] - 1))
            kernels_list.append(GPy.kern.RBF(1))

        lin_mf_kernel = kernels.LinearMultiFidelityKernel(kernels_list)
        gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=len(self.fidelities))

        # Fix noise terms for each fidelity
        for i,fidelity in enumerate(self.fidelities):
            # Construct the attribute name dynamically
            noise_attr = f"Gaussian_noise" if i == 0 else f"Gaussian_noise_{i}"
            try:
                getattr(gpy_lin_mf_model.mixed_noise, noise_attr).fix(self.noise[fidelity])
            except AttributeError:
                print(f"Error: Attribute '{noise_attr}' not found in the model.")
                raise


        # Wrap and optimize the model
        self.model = GPyMultiOutputWrapper(
            gpy_lin_mf_model, len(self.fidelities), n_optimization_restarts=n_restarts, verbose_optimization=True
        )
        self.model.optimize()
        return self.model

    def set_data(self,trainings_data_new):
        x_train = []
        y_train = []
        for fidelity in self.fidelities:
            self.trainings_data[fidelity][0].extend(trainings_data_new[fidelity][0])
            self.trainings_data[fidelity][1].extend(trainings_data_new[fidelity][1])
            x_tmp=np.atleast_2d(self.trainings_data[fidelity][0])
            y_tmp=np.atleast_2d(self.trainings_data[fidelity][1]).T
            x_train.append(x_tmp)
            y_train.append(y_tmp)
        
        X_train, Y_train = convert_xy_lists_to_arrays(x_train, y_train)
        self.model.set_data(X_train, Y_train)

    def max_acquisition_integrated_variance_reduction(self, parameters):
        ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
        spaces_tmp = []
        for i in parameters:
            spaces_tmp.append(ContinuousParameter(i, parameters[i][0], parameters[i][1]))
        
        spaces_tmp.append(InformationSourceParameter(self.nfidelities))
        parameter_space = ParameterSpace(spaces_tmp)

        optimizer = GradientAcquisitionOptimizer(parameter_space)
        multi_source_acquisition_optimizer = MultiSourceAcquisitionOptimizer(optimizer, parameter_space)
        #acquisition = ModelVariance(mf_model) * inequality_constraints
        acquisition = IntegratedVarianceReduction(self.model, parameter_space, num_monte_carlo_points=2000) * self.inequality_constraints

        # Create batch candidate point calculator
        sequential_point_calculator = SequentialPointCalculator(acquisition, multi_source_acquisition_optimizer)
        loop_state = create_loop_state(self.model.X, self.model.Y)
        x_next = sequential_point_calculator.compute_next_points(loop_state)

        return x_next, acquisition
    
    def max_acquisition_multisource(self, parameters):
        ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
        spaces_tmp = []
        for i in parameters:
            spaces_tmp.append(ContinuousParameter(i, parameters[i][0], parameters[i][1]))
        
        spaces_tmp.append(InformationSourceParameter(self.nfidelities))
        parameter_space = ParameterSpace(spaces_tmp)

        optimizer = GradientAcquisitionOptimizer(parameter_space)
        us_acquisition = MultiInformationSourceEntropySearch(self.model, parameter_space) * self.inequality_constraints
        x_new, _ = optimizer.optimize(us_acquisition)
        return x_new, us_acquisition
    
    def max_acquisition_model_variance(self, parameters):
        ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
        spaces_tmp = []
        for i in parameters:
            spaces_tmp.append(ContinuousParameter(i, parameters[i][0], parameters[i][1]))
        
        spaces_tmp.append(InformationSourceParameter(self.nfidelities))
        parameter_space = ParameterSpace(spaces_tmp)


        optimizer = GradientAcquisitionOptimizer(parameter_space)
        multi_source_acquisition_optimizer = MultiSourceAcquisitionOptimizer(optimizer, parameter_space)
        acquisition = ModelVariance(self.model) * self.inequality_constraints

        # Create batch candidate point calculator
        sequential_point_calculator = SequentialPointCalculator(acquisition, multi_source_acquisition_optimizer)
        loop_state = create_loop_state(self.model.X, self.model.Y)
        x_next = sequential_point_calculator.compute_next_points(loop_state)
        
        return x_next, acquisition

    def evaluate_model(self, x, fidelity=2):
        x_eval=np.array([x])
        SPLIT = 1
        X_eval = convert_x_list_to_array([x_eval , x_eval, x_eval])
        return self.model.predict(X_eval[int(fidelity)*SPLIT:int(fidelity+1)*SPLIT])[0][0][0]

    def evaluate_model_gradient(self, x, fidelity=2):
        x_eval=np.array([x])
        SPLIT = 1
        X_eval = convert_x_list_to_array([x_eval , x_eval, x_eval])
        return self.model.get_prediction_gradients(X_eval[int(fidelity)*SPLIT:int(fidelity+1)*SPLIT])[0][0]

    def evaluate_model_uncertainty(self, x, fidelity=2):
        x_eval=np.array([x])
        SPLIT = 1
        X_eval = convert_x_list_to_array([x_eval , x_eval, x_eval])
        _, var = self.model.predict(X_eval[int(fidelity)*SPLIT:int(fidelity+1)*SPLIT])
        var=var[0][0]
        var=np.sqrt(var)
        return var

    def get_min(self, parameters, x0=None, fidelity=2):

        def f(x):
            self.evaluate_model(x, fidelity)

        bnds=[]
        for i in parameters:
            bnds.append((parameters[i][0],parameters[i][1]))
            if x0==None:
                x0.append((parameters[i][1]-parameters[i][0])/2.)
        x0=np.array(x0)
        
        res = minimize(f, x0,bounds=bnds)
        return res.x, res.fun
    
    def get_min_constrained(self, parameters, fidelity=2):
        spaces_tmp = []
        for i in parameters:
            spaces_tmp.append(ContinuousParameter(i, parameters[i][0], parameters[i][1]))
        
        spaces_tmp.append(InformationSourceParameter(self.nfidelities))
        parameter_space = ParameterSpace(spaces_tmp)

        model = MFGPAuxilaryModel(self, fidelity)

        optimizer = GradientAcquisitionOptimizer(parameter_space)
        acquisition = model
        x_min, _ = optimizer.optimize(acquisition)
        x_min=[x for x in x_min[0]]
        return x_min, self.evaluate_model(x_min, fidelity)
    

class MFGPAuxilaryModel(Acquisition):
    def __init__(self, mf_model, fidelity):
        self.mf_model = mf_model
        self.fidelity = fidelity
        self.inequality=self.mf_model.inequality_constraints

    def evaluate(self, x):
        delta_inequ=self.inequality.evaluate(x)
        delta_inequ[delta_inequ == 0] = np.inf
        delta_x = np.ones(len(x))
        for i,xi in enumerate(x[:,:]):
            delta_x[i] = -1.*self.mf_model.evaluate_model(xi, self.fidelity)
            if self.mf_model.evaluate_model(xi, self.fidelity) <= 0.:
                delta_x[i] = -0.00001
        return delta_x[:, None]*delta_inequ[:,None]
    
    @property
    def has_gradients(self):
        return True
    
    def get_gradients(self,x):
        delta_x = np.ones(len(x))
        for i,xi in enumerate(x[:,:]):
            delta_x[i] = self.mf_model.evaluate_model_gradient(xi,self.fidelity)[0][0]
        return delta_x[:, None]

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x[:,:].shape)

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

        return delta_x[:, None]


    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x[:, :-1].shape)
