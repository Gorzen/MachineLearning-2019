import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

# Function to split data indices
# num_examples: total samples in the dataset
# k_fold: number fold of CV
# returns: array of shuffled indices with shape (k_fold, num_examples//k_fold)
def fold_indices(num_examples,k_fold):
    ind = np.arange(num_examples)
    split_size = num_examples//k_fold
    
    #important to shuffle your data
    np.random.shuffle(ind)
    
    k_fold_indices = []
    # Generate k_fold set of indices
    k_fold_indices = [ind[k*split_size:(k+1)*split_size] for k in range(k_fold)]
         
    return np.array(k_fold_indices)

'''Plotting Heatmap for CV results'''
def plot_cv_result(grid_val,grid_search_lambda,grid_search_degree):
    plt.figure(figsize=(8,10))
    plt.imshow(grid_val)
    plt.colorbar()
    plt.xticks(np.arange(len(grid_search_degree)), grid_search_degree, rotation=20)
    plt.yticks(np.arange(len(grid_search_lambda)), grid_search_lambda, rotation=20)
    plt.xlabel('degree')
    plt.ylabel('lambda')
    plt.title('Val Loss for different lambda and degree')
    plt.show()
    

'''
Grid Search Function
params:{'param1':[1,2,..,4],'param2':[6,7]} dictionary of search params
k_fold: fold for CV to be done
fold_ind: splits of training set
function: implementation of model should return a loss or score
X,Y: training examples
'''
def grid_search_cv(params,k_fold,fold_ind,function,X,Y):
    
    #might mess up with dictionary order
    param_grid = ParameterGrid(params)
    #save the values for the combination of hyperparameters
    grid_val = np.zeros(len(param_grid))
    grid_val_std = np.zeros(len(param_grid))   
    
    for i, p in enumerate(param_grid):
        print('Evaluating for {} ...'.format(p))
        loss = np.zeros(k_fold)
        for k in range(k_fold):
            loss[k] = function(k,fold_ind,X,Y,**p)
        grid_val[i] = np.mean(loss)
        grid_val_std[i] = np.std(loss)
    
    # reshape in the proper dimension of search space
    if len(params.keys())>1:
        search_dim = tuple([len(p) for _,p in params.items()])
        grid_val = grid_val.reshape(search_dim)
        grid_val_std = grid_val_std.reshape(search_dim)
    
    return grid_val, grid_val_std