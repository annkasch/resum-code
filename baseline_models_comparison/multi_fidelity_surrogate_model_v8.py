import numpy as np
np.random.seed(123)
import pandas as pd
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
from scipy.optimize import minimize, minimize_scalar

import GPy
import emukit.multi_fidelity
import emukit.test_functions
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.experimental_design.acquisitions import ModelVariance,IntegratedVarianceReduction
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.acquisition import Acquisition
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
from emukit.bayesian_optimization.acquisitions.expected_improvement import ExpectedImprovement
from emukit.core.loop.candidate_point_calculators import SequentialPointCalculator
import sys
from scipy.optimize import NonlinearConstraint
from emukit.core.loop.loop_state import create_loop_state
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
sys.path.append('../utilities')
import plotting_utils as plotting


# Construct a linear multi-fidelity model

def linear_multi_fidelity_model(X_train, Y_train , noise, n_fidelities, n_restarts=100):
    #X_train, Y_train = convert_xy_lists_to_arrays([x_train_lf,x_train_hf], [y_train_lf,y_train_hf])

    #kernels = [GPy.kern.RBF(X_train[0].shape[0]-1),GPy.kern.RBF(1)]
    kernels = [GPy.kern.RBF(X_train[0].shape[0]-1), GPy.kern.RBF(1)]
    
    for f in range(n_fidelities-2):
        kernels.append(GPy.kern.RBF(X_train[0].shape[0]-1))
        kernels.append(GPy.kern.RBF(1))
    

    lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=n_fidelities)

    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(noise[0])
    gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(noise[1])


    ## Wrap the model using the given 'GPyMultiOutputWrapper'
    lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, n_fidelities, n_optimization_restarts=n_restarts, verbose_optimization=True)

    ## Fit the model
    lin_mf_model.optimize()
    mf_model = lin_mf_model
    return mf_model

# Acqusition Curve

# Define cost of different fidelities as acquisition function
class Cost(Acquisition):
    def __init__(self, costs):
        self.costs = costs

    def evaluate(self, x):
        fidelity_index = x[:, -1].astype(int)
        x_cost = np.array([self.costs[i] for i in fidelity_index])
        return x_cost[:, None]
    
    @property
    def has_gradients(self):
        return True
    
    def evaluate_with_gradients(self, x):
        return self.evalute(x), np.zeros(x.shape)
    
class InequalityConstraints(Acquisition):
    def __init__(self):
        self.cx = 1.

    def evaluate(self, x):
        delta_x = np.ones(len(x))
        for i,xi in enumerate(x[:,:-1]):
            if plotting.get_inner_radius(xi) < 90.:
                delta_x[i] = 0.
            elif plotting.get_outer_radius(xi) > 265.:
                delta_x[i] = 0.
            elif plotting.get_outer_radius(xi)-plotting.get_inner_radius(xi)  > 20.:
                delta_x[i] = 0.
            elif xi[2]*xi[1]*xi[4] > 1.05 * np.pi*(plotting.get_outer_radius(xi)**2-plotting.get_inner_radius(xi)**2):
                delta_x[i] = 0.
            elif plotting.is_crossed(xi)==True:
                delta_x[i] = 0.
            else: 
                delta_x[i] =1.
        return delta_x[:, None]
    
    @property
    def has_gradients(self):
        return True
    
    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x[:,:-1].shape)
    
class MFModel(Acquisition):
    def __init__(self, mf_model, fidelity):
        self.mf_model = mf_model
        self.fidelity = fidelity

    def evaluate(self, x):
        delta_x = np.ones(len(x))
        for i,xi in enumerate(x[:,:]):
            delta_x[i] = -1.*evaluate_model(xi, self.mf_model, self.fidelity)
            if evaluate_model(xi, self.mf_model, self.fidelity) <= 0.:
                delta_x[i] = -0.00001
        return delta_x[:, None]
    
    @property
    def has_gradients(self):
        return True
    
    def get_gradients(self,x):
        delta_x = np.ones(len(x))
        for i,xi in enumerate(x[:,:]):
            delta_x[i] = evaluate_model_gradient(xi,self.mf_model, self.fidelity)[0][0]
        return delta_x[:, None]

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x[:,:].shape)
    
class inequality_constraints(Acquisition):
    def __init__(self):
        self.cx = 1.

    def evaluate(self, x):
        delta_x = np.ones(len(x))
        for i,xi in enumerate(x[:,:]):
            if plotting.get_inner_radius(xi) < 90.:
                delta_x[i] = np.inf
            elif plotting.get_outer_radius(xi) > 265.:
                delta_x[i] = np.inf
            elif plotting.get_outer_radius(xi)-plotting.get_inner_radius(xi)  > 20.:
                delta_x[i] = np.inf
            elif xi[2]*xi[1]*xi[4] > 1. * np.pi*(plotting.get_outer_radius(xi)**2-plotting.get_inner_radius(xi)**2):
                delta_x[i] = np.inf
            elif plotting.is_crossed(xi)==True:
                delta_x[i] = np.inf
            else: 
                delta_x[i] = 1.
        return delta_x[:, None]
    
    @property
    def has_gradients(self):
        return True
    
    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x[:,:].shape)

def get_min_constrained(mf_model, xlow, xhigh, labels, fidelity=2):
    spaces_tmp = []
    for i in range(len(labels)):
        spaces_tmp.append(ContinuousParameter(labels[i], xlow[i], xhigh[i]))
    
    parameter_space = ParameterSpace(spaces_tmp)
    ineq_constraints = inequality_constraints()
    model = MFModel(mf_model, fidelity)

    model = MFModel(mf_model, fidelity)
    optimizer = GradientAcquisitionOptimizer(parameter_space)
    acquisition = model * ineq_constraints
    x_min, _ = optimizer.optimize(acquisition)
    x_min=[x for x in x_min[0]]
    return x_min, evaluate_model(x_min, mf_model, fidelity)

def max_acquisition_multisource(mf_model, xlow, xhigh, labels):
    ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
    spaces_tmp = []
    for i in range(len(labels)):
        spaces_tmp.append(ContinuousParameter(labels[i], xlow[i], xhigh[i]))
    
    spaces_tmp.append(InformationSourceParameter(3))
    parameter_space = ParameterSpace(spaces_tmp)

    inequality_constraints = InequalityConstraints()
    optimizer = GradientAcquisitionOptimizer(parameter_space)
    us_acquisition = MultiInformationSourceEntropySearch(mf_model, parameter_space) * inequality_constraints
    x_new, _ = optimizer.optimize(us_acquisition)
    return x_new, us_acquisition

def max_acquisition_model_variance(mf_model, xlow, xhigh, labels):
    ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
    spaces_tmp = []
    for i in range(len(labels)):
        spaces_tmp.append(ContinuousParameter(labels[i], xlow[i], xhigh[i]))
    
    spaces_tmp.append(InformationSourceParameter(3))
    parameter_space = ParameterSpace(spaces_tmp)
    inequality_constraints = InequalityConstraints()

    optimizer = GradientAcquisitionOptimizer(parameter_space)
    multi_source_acquisition_optimizer = MultiSourceAcquisitionOptimizer(optimizer, parameter_space)
    acquisition = ModelVariance(mf_model) * inequality_constraints

    # Create batch candidate point calculator
    sequential_point_calculator = SequentialPointCalculator(acquisition, multi_source_acquisition_optimizer)
    loop_state = create_loop_state(mf_model.X, mf_model.Y)
    x_next = sequential_point_calculator.compute_next_points(loop_state)
    
    return x_next, acquisition

def max_acquisition_integrated_variance_reduction(mf_model, xlow, xhigh, labels):
    ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
    spaces_tmp = []
    for i in range(len(labels)):
        spaces_tmp.append(ContinuousParameter(labels[i], xlow[i], xhigh[i]))
    
    spaces_tmp.append(InformationSourceParameter(3))
    parameter_space = ParameterSpace(spaces_tmp)
    inequality_constraints = InequalityConstraints()

    optimizer = GradientAcquisitionOptimizer(parameter_space)
    multi_source_acquisition_optimizer = MultiSourceAcquisitionOptimizer(optimizer, parameter_space)
    #acquisition = ModelVariance(mf_model) * inequality_constraints
    acquisition = IntegratedVarianceReduction(mf_model, parameter_space, num_monte_carlo_points=2000) * inequality_constraints

    # Create batch candidate point calculator
    sequential_point_calculator = SequentialPointCalculator(acquisition, multi_source_acquisition_optimizer)
    loop_state = create_loop_state(mf_model.X, mf_model.Y)
    x_next = sequential_point_calculator.compute_next_points(loop_state)

    return x_next, acquisition
'''
def max_acquisition_expected_improvement(mf_model, xlow, xhigh, labels):
    ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
    spaces_tmp = []
    for i in range(len(labels)):
        spaces_tmp.append(ContinuousParameter(labels[i], xlow[i], xhigh[i]))
    
    spaces_tmp.append(InformationSourceParameter(2))
    parameter_space = ParameterSpace(spaces_tmp)
    inequality_constraints = InequalityConstraints()

    optimizer = GradientAcquisitionOptimizer(parameter_space)
    multi_source_acquisition_optimizer = MultiSourceAcquisitionOptimizer(optimizer, parameter_space)
    #acquisition = ModelVariance(mf_model) * inequality_constraints
    acquisition = ExpectedImprovement(mf_model) * inequality_constraints

    # Create batch candidate point calculator
    sequential_point_calculator = SequentialPointCalculator(acquisition, multi_source_acquisition_optimizer)
    loop_state = create_loop_state(mf_model.X, mf_model.Y)
    x_next = sequential_point_calculator.compute_next_points(loop_state)
    
    return x_next, acquisition
'''

def add_sample(x_train, y_train, mode, labels, ylabel,sample, version='v1'):
    data=pd.read_csv(f'in/Ge77_rates_new_samples_{version}.csv')
    row=list(set(data.index[data['Sample'] == sample].tolist()).intersection(set(data.index[data['Mode'] == mode].tolist())))
    if mode == 1.:
        fidelity="HF"
    else:
        fidelity="LF"
    for i in row:
        x_new=[]
        for l in labels:
            x_new.append(data.iloc[i][l])
        x_train=np.append(x_train,np.array([x_new]),axis=0)
        y_train=np.append(y_train,np.array([[data.iloc[i][ylabel]]]),axis=0)
        print(f"Adding {fidelity} sample at {x_new} with {ylabel} of {y_train[-1]}")
    return x_train, y_train

def add_samples(x_train_l, y_train_l, x_train_h, y_train_h, mf_model, labels, ylabel, sample, version='v1'):
    x_train_l, y_train_l = add_sample(x_train_l, y_train_l, 0.0, labels, ylabel, sample, version)
    x_train_h, y_train_h = add_sample(x_train_h, y_train_h, 1.0, labels, ylabel,sample, version)
    
    X_train, Y_train = convert_xy_lists_to_arrays([x_train_l, x_train_h], [y_train_l, y_train_h])
    mf_model.set_data(X_train, Y_train)
    return x_train_l, y_train_l, x_train_h, y_train_h, mf_model

def get_num_new_samples(version='v1'):
    data=pd.read_csv(f'in/Ge77_rates_new_samples_{version}.csv')
    if len(data.loc[data['Mode'] == 1.]["Sample"].to_numpy())==0:
        nsamples_hf = -1
    else: 
        nsamples_hf=np.max(data.loc[data['Mode'] == 0.]["Sample"].to_numpy())
    if len(data.loc[data['Mode'] == 1.]["Sample"].to_numpy())==0:
        nsamples_lf = -1
    else: 
        nsamples_lf=np.max(data.loc[data['Mode'] == 0.]["Sample"].to_numpy())
    return [nsamples_lf,nsamples_hf]

# Get an evaluation of the HF model prediction at a certain point

def evaluate_model(x, mf_model, fidelity=2):
    x_eval=np.array([x])
    SPLIT = 1
    X_eval = convert_x_list_to_array([x_eval , x_eval, x_eval])
    return mf_model.predict(X_eval[int(fidelity)*SPLIT:int(fidelity+1)*SPLIT])[0][0][0]

def evaluate_model_gradient(x, mf_model, fidelity=2):
    x_eval=np.array([x])
    SPLIT = 1
    X_eval = convert_x_list_to_array([x_eval , x_eval, x_eval])
    return mf_model.get_prediction_gradients(X_eval[int(fidelity)*SPLIT:int(fidelity+1)*SPLIT])[0][0]

def evaluate_model_uncertainty(x, mf_model, fidelity=2):
    x_eval=np.array([x])
    SPLIT = 1
    X_eval = convert_x_list_to_array([x_eval , x_eval, x_eval])
    _, var = mf_model.predict(X_eval[int(fidelity)*SPLIT:int(fidelity+1)*SPLIT])
    var=var[0][0]
    var=np.sqrt(var)
    return var

# Get the minimum of the HF model prediction

def get_min(mf_model, xmin, xmax, fidelity=2):

    def f(x):
        evaluate_model(x, mf_model, fidelity)

    x0=np.array([150.,10.,360.,0.,4.])
    bnds=[]
    for i in range(len(xmin)):
        bnds.append((xmin[i],xmax[i]))
    
    res = minimize(f, x0,bounds=bnds)
    return res.x, res.fun

def get_min_constrained_scipy(mf_model, xmin, xmax, fidelity=2):

    con1 = lambda x: 265.-plotting.get_outer_radius(x)
    con2 = lambda x: plotting.get_inner_radius(x) - 90.
    con3 = lambda x: 20.-plotting.get_outer_radius(x)-plotting.get_inner_radius(x)
    con4 = lambda x: 1.05-(x[2]*x[1]*x[4])/(np.pi*(plotting.get_outer_radius(x)**2-plotting.get_inner_radius(x)**2))
    
    nlc1 = NonlinearConstraint(con1, 0, 1)
    nlc2 = NonlinearConstraint(con2, 0, 1)
    nlc3 = NonlinearConstraint(con3, 0, 1)
    nlc4 = NonlinearConstraint(con4, 0, 1)

    def f(x):
        evaluate_model(x, mf_model, fidelity)
    
    x0=np.array([150,10,360,0.,5.])

    bnds=[]
    for i in range(len(xmin)):
        bnds.append((xmin[i],xmax[i]))
    np.random.seed(0)
    
    opt = {'verbose': 1,'disp':False,'maxiter': 2000}
    method = 'trust-constr'
    res = minimize(f, x0, method=method, constraints=[nlc1, nlc2,nlc3, nlc4],options=opt, bounds=bnds)

    return res.x, res.fun

def check_inequalities(x):
            if plotting.get_inner_radius(x) <= 90.:
                return False
            elif plotting.get_outer_radius(x) >= 265.:
                return False
            elif plotting.get_outer_radius(x)-plotting.get_inner_radius(x)  > 20.:
                return False
            elif (x[2]*x[1]*x[4])/(np.pi*(plotting.get_outer_radius(x)**2-plotting.get_inner_radius(x)**2)) > 1.:
                return False
            else: 
                return True

