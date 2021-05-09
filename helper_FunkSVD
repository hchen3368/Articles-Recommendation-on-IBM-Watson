import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
%matplotlib inline

def FunkSVD(ratings_mat, latent_features=4, learning_rate=0.0001, iters=100):
    '''
    This function performs matrix factorization using a basic form of FunkSVD with no regularization

    INPUT:
    ratings_mat - (numpy array) a matrix with users as rows, items as columns, and ratings as values
    latent_features - (int) the number of latent features used
    learning_rate - (float) the learning rate
    iters - (int) the number of iterations

    OUTPUT:
    user_mat - (numpy array) a user by latent feature matrix
    item_mat - (numpy array) a latent feature by item matrix
    '''

    # Set up useful values to be used through the rest of the function
    n_users = ratings_mat.shape[0]
    n_items = ratings_mat.shape[1]
    num_ratings = np.sum(~np.isnan(ratings_mat)) # total number of ratings in the matrix

    # initialize the user and item matrices with random values

    user_mat = np.random.rand(n_users, latent_features)
    item_mat = np.random.rand(latent_features, n_items)

    # initialize sse at 0 for first iteration
    sse_accum = 0

    # header for running results
    print("Optimization Statistics")
    print("Iterations | Mean Squared Error ")

    for i in range(iters):

        # update our sse
        old_sse = sse_accum
        sse_accum = 0

        # compute the error as the actual minus the dot product of the user and item latent features
        error = np.nan_to_num(ratings_mat-user_mat@item_mat)
        # Keep track of the total sum of squared errors for the matrix
        sse_accum = np.sum(np.multiply(error,error))
        # update the values in each matrix in the direction of the gradient
        user_mat += 2*learning_rate*error@(item_mat.T)
        item_mat += 2*learning_rate*(user_mat.T)@error
        # print results for iteration
        print(f'Iteration {i}, Sum of Square Errors is {sse_accum}.')

    return user_mat, item_mat 
