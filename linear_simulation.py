#Synthetic data simulation of scoring rule ensembles
import datetime as dt
import multiprocessing as mp
import numpy as np
from numpy.linalg import eig, inv
import pandas as pd
from scipy.stats import bernoulli
import sys
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import time

# importing machine learning models for prediction
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression


def generate_linear_data(n, beta, sigma, eps_sigma, categorical=True, mean = None):
    '''Generate linear categorical data from beta and sigma
    n = number of x, y pairs to generate
    beta = vector of linear constants: y = beta * x + epsilon
    sigma = covariance matrix of x
    eps_sigma = standard deviation of epsilon noise
    categorical = boolean variable; values are categorical if True, continuous if False
    mean = list of length dim(x) when user wishes to specify the means of each component of x. If None, program sets to vector of 0s.
    '''
    if type(mean) != list:
        mean = [0]*sigma.shape[0] #creates vector of zeros to serve as means of each component of x`
    elif len(mean) != len(beta):
        sys.exit('Length of list mean list is different than number of coefficients in beta')

    x = np.random.multivariate_normal(mean, sigma, size=n)
    if categorical == True:
        beta_x = np.matmul(np.asarray(beta), x.T)
        sigmoid = np.where(beta_x < 0, np.exp(beta_x)/(1 + np.exp(beta_x)), \
                    1/(1 + np.exp(-beta_x)))
        y = np.where(np.random.random((n,)) < sigmoid, 1, 0)
        '''
        print('x',x)
        print('beta',beta)
        print('beta_x',beta_x)
        print('sigmoid',sigmoid)
        print('y',y)
        '''
    else:
        y = np.matmul(np.asarray(beta), x.T) + np.random.normal(0, eps_sigma, size=n)
    return x, y


def cov_shifter(cov, covariance_shift_stdev):
    #This function takes a covariance matrix and a shift coefficient, and shifts it!
    eig_vals, eig_vecs = eig(cov)
    noise = np.random.normal(1, covariance_shift_stdev, (len(eig_vals),1))
    eig_vals_mod = np.multiply(eig_vals, noise.T)[0] #[0] gives a vector instead of a 2-d matrix
    tmp = np.matmul(eig_vecs, np.diag(eig_vals_mod))
    cov2 = np.matmul(tmp, inv(eig_vecs))
    #B = np.random.rand(cov.shape[0], cov.shape[0])
    #cov_additive = np.dot(B, B.transpose())
    #cov2 = cov2 + cov_additive*.1
    return cov2


def calc_proper_loss(weights, predictions, y, scoring_rule_type = 'quadratic'):
    '''This function takes a list of weigts, an array of model predictions, and a set
    of true y values, and it calculates a negative scoring rule loss
    '''
    #0 Aggregate Predictions
    weights = [np.abs(i) / np.sum(np.abs(weights)) for i in weights] #normalize and enforce positivity
    wtd_predictions = np.matmul(weights, predictions)
    #print('weights',np.round(weights,2))

    #1 Calculate Proper Loss
    if scoring_rule_type == 'quadratic':
        #calculate the proper proper
        score = np.sum([2*i*j + 2*(1-i)*(1-j) for i,j in zip(wtd_predictions, y)])\
                    - np.sum(np.square(wtd_predictions)) - \
                    np.sum(np.square(1-wtd_predictions))

    else:
        sys.exit('Sorry, weve only built the quadratic scoring rule so far.')

    #print('proper_loss: ', np.round(-score,2))
    return -score


def calc_sse_loss(weights, predictions, y):
    '''This function takes a list of weigts, an array of model predictions, and a set
    of true y values, and it calculates the sum of squared errors
    '''
    weights = [np.abs(i) / np.sum(np.abs(weights)) for i in weights] #normalize and enforce positivity
    wtd_predictions = np.matmul(weights, predictions)
    return np.sum([(i - j)**2 for i,j in zip(wtd_predictions, y)])


def vector_norm_shift(vec, shift):
    #This function takes a per-element shift in vector norm and performs additive shifts to acheive it
    if len(vec.shape) == 1: vec = vec.reshape((1,vec.shape[0]))
    shift = shift*vec.shape[1] #scale per-dimension shift to total shift
    norm_vec = np.linalg.norm(vec)
    shifts_vec = np.random.random(vec.shape)
    shifts_vec = (shifts_vec / np.sum(shifts_vec))*shift**2 #normalize shifts to add to 1, then multiply by total amount of shift
    rand_signs = (2*np.random.randint(0,2,size=(vec.shape[1]))-1)
    shifted_vec = vec + np.multiply(np.sqrt(shifts_vec), rand_signs)
    '''
    #test that total shift is what we wanted:
    print('shift', shift)
    diffs = vec - shifted_vec
    print('diffs', diffs)
    print('norm', np.linalg.norm(diffs))
    print('vec', vec)
    '''
    return shifted_vec


def kl_mvn(m0, S0, m1, S1):
    """
    Inspiration / Source: https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    #return .5 * (tr_term + det_term + quad_term - N)
    return .5 * (tr_term + det_term + quad_term - N)

def evaluate_alpha(alpha, weights, predictions, y):
    '''This function takes an alpha value, a list of weights, an array of model predictions, and a set
    of true y values, and it calls another function to calculate the proper loss
    '''
    alpha = np.clip(alpha, 0, 1)[0]
    loss = calc_proper_loss([alpha*i + (1-alpha)/len(weights) for i in weights], + \
                predictions, y, scoring_rule_type = 'quadratic') #runs proper loss with new weights which are alpha * weights + (1-alpha) * uniform_weights
    return loss

def main(ensemble_fraction_of_train = .2,n_models = 10, n_deploy = 100, \
                dim_x = 10, beta_norm_shift = .05, covariance_shift_stdev = .4,
                alpha = 1.):
    '''This function generates random data, fits a set of models to it, generates
    a shifted version of this data, then optimizes the weightings of the ensembles
    based on a limted number of draws from this shifted distribution.
    Variables passed in:
    - ensemble_fraction_of_train = percentage of original training set data used to produce each ensemble member
    - n_models = number of models to fit and potentially include in ensembles
    - n_deploy = number of ground truth samples from deployment on which we can optimize our ensemble
    - dim_x = the number of x variables in a given training sample
    - beta_norm_shift = per-element shift in the norm of the beta vector defining the linear generating model
    - covariance_shift_stdev = standard deviation of the shifts in eigenvalues of the new cov2 covariance matrix relative to cov
    - alpha = fraction of ensemble model weights determined by MLE of proper loss function, with (1-alpha) given to uniform weights
    '''

    #0 Setup
    n = 1000 #number of training data points to generate
    n_deploy_oos = 10**3 #number of out of sample deployment samples to use for testing
    A = np.random.rand(dim_x, dim_x)
    cov = np.dot(A, A.transpose())
    beta = np.random.normal(0,.3,dim_x)
    eps_sigma = .4

    #1 Make permuted covariance matrix
    cov2 = cov_shifter(cov, covariance_shift_stdev)

    #2 Generate n datapoints & fit model(s)
    x, y = generate_linear_data(n, beta, cov, eps_sigma, categorical = True)
    #Some credit for the below goes to: https://www.geeksforgeeks.org/ensemble-methods-in-python/

    # Splitting between train data into training and validation dataset
    models = [] #list which will contain model classes
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    for i in range(n_models): # Make temporary subsets of the training set and fit an ensemble member from each
        #model_init = RandomForestRegressor() #unfitted model which will be fit on different training sets
        model_init = RandomForestRegressor()
        x_train_tmp, x_test_tmp, y_train_tmp, y_test_tmp = \
                    train_test_split(x_train, y_train, test_size=\
                    (1-ensemble_fraction_of_train), random_state=\
                    int(100*np.random.random()), shuffle=True)
        models += [model_init.fit(x_train_tmp, y_train_tmp)]

    '''
    # Other model types
    model_1 = LinearRegression()
    model_2 = xgb.XGBRegressor()
    model_3 = RandomForestRegressor()
    model_4 = LogisticRegression()
    '''

    #3 Generate n_test datapoints from "real world" distribution
    beta2 = vector_norm_shift(beta.copy(), beta_norm_shift)
    x_deploy, y_deploy = generate_linear_data(n_deploy, beta2, cov2, eps_sigma, \
                categorical = True)

    #4 Optimize ensemble weights to minimize losses
    #Start with basic minimizer of linear weighting
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    #All model predictions will be a single number, as we only allow for binary classification or regression
    predictions_test = np.asarray([i.predict(x_test) for i in models])
    predictions_deploy = np.asarray([i.predict(x_deploy) for i in models])#np.asarray([model_1.predict(x_deploy), model_2.predict(x_deploy), model_3.predict(x_deploy)])
    #print('predictions_deploy', predictions_deploy)
    weights_static_proper = minimize(calc_proper_loss, [1./n_models]*n_models, \
                args=(predictions_test, y_test), method='Powell').x
    weights_static_sse = minimize(calc_sse_loss, [1./n_models]*n_models, \
                args=(predictions_test, y_test), method='Powell').x
    weights_deploy_proper = minimize(calc_proper_loss, [1./n_models]*n_models, \
                args=(predictions_deploy, y_deploy), method='Powell').x
    weights_deploy_sse = minimize(calc_sse_loss, [1./n_models]*n_models, \
                args=(predictions_deploy, y_deploy), method='Powell').x

    weights_static_proper = [np.abs(i) / np.sum(np.abs(weights_static_proper)) for i in weights_static_proper] #normalize and enforce positivity
    weights_static_sse = [np.abs(i) / np.sum(np.abs(weights_static_sse)) for i in weights_static_sse] #normalize and enforce positivity
    weights_deploy_proper = [np.abs(i) / np.sum(np.abs(weights_deploy_proper)) for i in weights_deploy_proper] #normalize and enforce positivity
    weights_deploy_sse = [np.abs(i) / np.sum(np.abs(weights_deploy_sse)) for i in weights_deploy_sse] #normalize and enforce positivity

    #5 Draw new samples, calculate and print results
    #5.1 Create deployment data
    x_deploy_oos, y_deploy_oos = generate_linear_data(n_deploy_oos, beta2, cov2, eps_sigma, \
                categorical = True) #Out of sample deployment data
    predictions_deploy_oos = np.asarray([i.predict(x_deploy_oos) for i in models])#np.asarray([model_1.predict(x_deploy), model_2.predict(x_deploy), model_3.predict(x_deploy)])

    #5.2 Calculate ground truth best weights on out of sample data
    weights_ground_truth = minimize(calc_proper_loss, [1./n_models]*n_models, \
                args=(predictions_deploy_oos, y_deploy_oos), method='Powell').x

    #5.3 Calculate optimal alpha on out of sample data
    optimal_alpha = np.clip(minimize(evaluate_alpha, .5, args=(weights_deploy_proper, \
                predictions_deploy_oos, y_deploy_oos)).x, 0, 1)[0]
    weights_alpha = [optimal_alpha*i + (1-optimal_alpha)/dim_x for i in weights_deploy_proper]

    #5.4 Build results list to be returned by main() function
    performance = []
    performance += [np.round(calc_proper_loss(weights_static_proper, predictions_deploy_oos, \
                y_deploy_oos),2)]
    performance += [np.round(calc_sse_loss(weights_static_proper, predictions_deploy_oos, \
                y_deploy_oos),2)]
    performance += [np.round(calc_proper_loss(weights_static_sse, predictions_deploy_oos, \
                y_deploy_oos),2)]
    performance += [np.round(calc_sse_loss(weights_static_sse, predictions_deploy_oos, \
                y_deploy_oos),2)]
    performance += [np.round(calc_proper_loss(weights_deploy_proper, predictions_deploy_oos,\
                y_deploy_oos),2)]
    performance += [np.round(calc_sse_loss(weights_deploy_proper, predictions_deploy_oos, \
                y_deploy_oos),2)]
    performance += [np.round(calc_proper_loss(weights_deploy_sse, predictions_deploy_oos,\
                y_deploy_oos),2)]
    performance += [np.round(calc_sse_loss(weights_deploy_sse, predictions_deploy_oos, \
                y_deploy_oos),2)]
    performance += [np.round(calc_proper_loss([1/n_models]*n_models, predictions_deploy_oos,\
                y_deploy_oos),2)]
    performance += [np.round(calc_sse_loss([1/n_models]*n_models, predictions_deploy_oos, \
                y_deploy_oos),2)]
    performance += [np.round(calc_proper_loss(weights_ground_truth, predictions_deploy_oos,\
                y_deploy_oos),2)]
    performance += [np.round(calc_sse_loss(weights_ground_truth, predictions_deploy_oos, \
                y_deploy_oos),2)]
    performance += [np.round(calc_proper_loss(weights_alpha, predictions_deploy_oos,\
                y_deploy_oos),2)]
    performance += [np.round(calc_sse_loss(weights_alpha, predictions_deploy_oos, \
                y_deploy_oos),2)]
    performance += [performance[8] - performance[4]]
    performance += [performance[10] - performance[4]]
    performance += [performance[12] - performance[4]]
    performance += [kl_mvn(np.asarray([0]*dim_x), cov, np.asarray([0])*dim_x, cov2)] #KL divergence between training and deployment data

    #6 Calculate performance for each individual model
    #zeros = np.zeros((1,n_models))
    #model_by_model = [np.round(calc_proper_loss([0,0,1], predictions_deploy_oos, y_deploy_oos),2) for i in range(n_models)]

    #print('performance_list: ', performance_list + ground_truth_results)

    print('optimal_alpha', optimal_alpha)

    return performance, [ensemble_fraction_of_train, n_models, n_deploy, dim_x,\
                beta_norm_shift, covariance_shift_stdev, n, eps_sigma, optimal_alpha] #second list passes settings


def simulation_loop(multiprocessing=False, ensemble_fraction_of_train_vec=None, \
            n_models_vec=None, n_deploy_vec=None, dim_x_vec=None, \
            beta_norm_shift_vec=None, covariance_shift_stdev_vec=None):
    '''
    This function varies a parameter of interest, calls main, and stores results
    in a .csv file
    '''
    #0 Set parameter values to loop through
    n_reps = 50
    if multiprocessing == False: #set parameter values here rather than having them passed into simulation_loop()
        vector_len = 9
        ensemble_fraction_of_train_vec = [.1, .2, .3, .4, .5, .6, .7, .8, .9] #  [.1]*vector_len # [.01, .025, .05, .075, .1, .15, .2]#
        n_models_vec = [20]*vector_len #[2, 4, 6, 8, 10, 15, 20]
        n_deploy_vec = [100]*vector_len #[5, 25, 125, 625]#[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] #[1000]*vector_len #
        dim_x_vec = [10]*vector_len #[2,4,6,8,10,12,14,16,18,20] #
        beta_norm_shift_vec = [0]*vector_len#[0, .05, .1, .15, .2, .25, .3]#
        covariance_shift_stdev_vec = [0]*vector_len #[1.]*vector_len #[.2, .4, .6, .8, 1, 2]

    #1 Run main for each set of parameters and write to .csv file
    for j in range(len(n_deploy_vec)):
        performance_list = []
        for rep in range(n_reps):
            print('Configuration: ', j, 'Rep: ', rep)
            performance, settings = main(ensemble_fraction_of_train=ensemble_fraction_of_train_vec[j],\
                        n_models=n_models_vec[j],\
                        n_deploy = n_deploy_vec[j], \
                        dim_x = dim_x_vec[j], \
                        beta_norm_shift=beta_norm_shift_vec[j],\
                        covariance_shift_stdev=covariance_shift_stdev_vec[j]
            )
            performance_list += [performance]

        row = '\n' + str(time.strftime('%Y%m%d')) +','+ str(n_reps)
        for setting in settings:
            row += ','
            row += str(setting)
        performance_list = np.round(np.mean(np.asarray(performance_list), axis=0),2)
        for k in range(performance_list.shape[0]):
            row += ','
            row += str(performance_list[k])

        with open('ensemble_scoring_simulations.csv','a') as fd:
            fd.write(row)
        if multiprocessing == True: print('One configuration just printed!')
    return

simulation_loop()

def run_simulation_loop_parallel():
    '''This function runs the simulation_loop function on various settings in parallel.
    Each processor runs simulation_loop with a different set of parameters and writes
    the results to ensemble_scoring_simulations.csv
    #Note that inputs must be a list of tuples
    '''
    #0 Lists of configurations
    vector_len = 7
    ensemble_fraction_of_train_vec = [.1]*vector_len*2 #[.1, .2, .3, .4, .5, .6, .7, .8, .9] #   [.01, .025, .05, .075, .1, .15, .2]#
    n_models_vec = [20]*vector_len*2 #[2, 4, 6, 8, 10, 15, 20, 50, 100] #
    n_deploy_vec = [100]*vector_len*2 #[10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]*2 #[5, 25, 125, 625]#[1000]*vector_len #
    dim_x_vec = [20]*vector_len*2 #[2,4,6,8,10,12,14,16,18,20] #
    beta_norm_shift_vec = [0, .05, .1, .15, .2, .25, .3] + [.1]*vector_len #[0]*vector_len  #
    covariance_shift_stdev_vec = [.4]*vector_len + [0, .2, .4, .6, .8, 1, 2] #[0]*vector_len + [.4]*vector_len +  #[1.]*vector_len #

    #1 Make inputs list of tuples
    inputs = []
    for i in range(len(ensemble_fraction_of_train_vec)):
        inputs += [(True, [ensemble_fraction_of_train_vec[i]], [n_models_vec[i]],\
                    [n_deploy_vec[i]], [dim_x_vec[i]], [beta_norm_shift_vec[i]],\
                    [covariance_shift_stdev_vec[i]],)]

    #2 Run simulation loop in parallel
    p = mp.Pool(4) #number of cores on my MacBook Air
    p.starmap(simulation_loop, inputs)

    return

run_simulation_loop_parallel()


'''
dim_x = 10
A = np.random.rand(dim_x, dim_x)
cov = np.dot(A, A.transpose())
B = np.random.rand(dim_x, dim_x)
cov2 = np.dot(B, B.transpose())

cov3 = cov_shifter(cov, 10)

kl_divergence = kl_mvn(np.asarray([0]*dim_x), cov, np.asarray([0]*dim_x), cov2)
print('kl divergence', kl_divergence)
print('kl divergence 2', kl_mvn(np.asarray([0]*dim_x), cov, np.asarray([0]*dim_x), cov3))
'''
