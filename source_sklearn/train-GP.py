from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.externals import joblib

## TODO: Import any additional libraries you need to define a model

## We are going to use a classifier based on Gaussian Processes
from sklearn.gaussian_process import GaussianProcessClassifier
import sklearn.gaussian_process.kernels as gpkernels
from sklearn import metrics

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model

## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--kernel', type=str, default='RBF')
    parser.add_argument('--length_scale', type=float, default=1.0)
    parser.add_argument('--length_scale_lower_bound', type=float, default=1e-5)
    parser.add_argument('--length_scale_upper_bound', type=float, default=1e5)
    parser.add_argument('--alpha',type=float, default=1.5)
    parser.add_argument('--alpha_lower_bound', type=float, default=1e-5)
    parser.add_argument('--alpha_upper_bound', type=float, default=1e5)
    parser.add_argument('--multiplier', type=float, default=1.0)
    parser.add_argument('--bias', type=float, default=0.0)

    
    ## TODO: Add any additional arguments that you will need to pass into your model
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    
    length_scale = args.length_scale
    length_scale_lb = args.length_scale_lower_bound
    length_scale_ub = args.length_scale_upper_bound
    alpha_lb = args.alpha_lower_bound
    alpha_ub = args.alpha_upper_bound
    alpha = args.alpha
    nu = args.alpha
    kernel = args.kernel
    
    multiplier = 1.0
    bias = args.bias
    
    kernels =  ['RBF', 'Matern', 'RationalQuadratic']
    assert kernel in kernels
    if kernel == 'RBF':
        print('RBF model')
        kernelfun = multiplier*gpkernels.RBF(length_scale, 
                                             length_scale_bounds = (length_scale_lb, length_scale_ub))
    elif kernel == 'Matern':
        print('Matern')
        kernelfun = multiplier*gpkernels.Matern(length_scale, nu = nu, 
                                                length_scale_bounds = (length_scale_lb, length_scale_ub))
    elif kernel == 'RationalQuadratic':
        print('RationalQuadratic')
        kernelfun = multiplier*gpkernels.RationalQuadratic(length_scale, alpha = alpha, 
                                                length_scale_bounds = (length_scale_lb, length_scale_ub), 
                                                alpha_bounds = (alpha_lb, alpha_ub))
    else:
        print('It should have not reached here!')
        kernelfun = 1.0*gpkernels.RBF(1.0)
    
    #RBF, Matern, ConstantKernel, WhiteKernel, RationalQuadratic
    # length_scale=1.0, length_scale_bounds=(1e-05, 100000.0), nu=1.5
    # length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-05, 100000.0), alpha_bounds=(1e-05, 100000.0)
    
    #device = torch.device("cpu")

    ## TODO: Define a model 
    #print('We instiantate the model')
    model = GaussianProcessClassifier(kernelfun)

    ## TODO: Train the model
    #print('We fit the model')
    
    model.fit(train_x, train_y)
   
    print('score-training {}'.format(model.score(train_x,train_y)))
          
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
