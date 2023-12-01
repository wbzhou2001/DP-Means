import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

class dpmeans(object):
    '''
    Python implementation of the dpmeans algorithm.
    Code modified from BAYESIAN LEARNING OF PLAY STYLES IN MULTIPLAYER VIDEO GAMES (2015) by Normoyle et.al.
    '''
    def __init__(self, data : np.array, lam : float):
        '''
        data : [batch, data_dimension] np.array
        lam : threshold for clustering
        '''
        assert len(data.shape) == 2, 'Check data dimension!'
        self.data = data
        self.lam = lam
        self.k = 0
        x_rand = self.data[np.random.choice(self.data.shape[0])]
        self.mu = np.array([x_rand])
        self.S = [[]]
    
    def train(self, epochs : int):
        '''
        Exit criterion is by number of epochs.
        '''
        for epoch in range(epochs):
            self.S = [[] for i in range(self.k + 1)]    # initialize new set of cluster records (because we are going over all data again)
            X_perm = self.data[np.random.permutation(self.data.shape[0])]
            # -------- cluster assignments --------
            for x in X_perm:
                c = self.argmin_l2(x, self.mu)
                if np.linalg.norm(x - self.mu[c]) > self.lam**2:
                    self.k += 1
                    self.S.append([])                   # [ k + 1 ]
                    self.mu = self.extend_mu(self.mu)   # [ k + 1 ]
                    self.S[self.k].append(x)            # new element in new cluster
                    self.mu[self.k] = x                 # new centroid
                else:
                    self.S[c].append(x)
            # -------- centroid updates --------
            for i in range(self.k):
                self.mu[i] = np.mean(self.S[i], 0)
        return self.S, self.k
    
    @staticmethod
    def argmin_l2(x : np.array, mu : np.array):
        xlist = np.array([x for i in range(len(mu))])
        rec = [np.linalg.norm(xlist - mui) for mui in mu]
        index = np.where(rec == min(rec))[0][0]         # get int
        return index
    
    @staticmethod
    def extend_mu(mu : np.array):
        temp = list(mu)
        temp.append(np.zeros_like(mu[0]))
        return np.array(temp)
    
    @staticmethod
    def plot_2d(S : list):
        '''
        Visualize the 2D plot of the cluster on the dataset
        '''
        colors = ['green', 'yellow', 'red']
        norm = Normalize(vmin=0., vmax=1.)
        linear_cmap = LinearSegmentedColormap.from_list('Cluster ID', colors= colors, N = 18)

        for cluster_id in range(len(S)):
            for pt_id in range(len(S[cluster_id])):
                plt.scatter(S[cluster_id][pt_id][0], S[cluster_id][pt_id][1], c = cluster_id / len(S), cmap = linear_cmap, norm=norm, s = 100)
        plt.grid()
        plt.title('DP-Means')
        plt.show()
        return None
