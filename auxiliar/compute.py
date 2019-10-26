import pickle
import tensorflow as tf
import numpy as np
import scipy
import scipy.io as sio
# from scipy.stats import multivariate_normal
from sys import version_info, argv
from datetime import datetime




def compute_rdm(data, measure='correlation'):
    return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data,measure))


def compute_mds(rdm, ndims=2):
    embedding = MDS(n_components=ndims, metric=True, dissimilarity='precomputed')
    return embedding.fit_transform(rdm)


def compute_respmat(x, y, ctx=[]):
    idx = 0
    if len(ctx):
        respmat = np.empty([16,x.shape[1]])
        for ii in range(2):
            for jj in range(8):
                respmat[idx,:] = np.mean(x[(y==jj+1) & (ctx==ii),0:],axis=0)
                idx +=1
    else:
        respmat = np.empty([8,x.shape[1]])
        for jj in range(8):
            respmat[idx,:] = np.mean(x[(y==jj+1),0:],axis=0)
            idx +=1
    return respmat
