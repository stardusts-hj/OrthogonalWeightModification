"""
Timo Flesch, 2019
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy
# %%

def plot_lcurves(loss,acc):
    plt.figure()
    plt.rcParams["figure.figsize"] = (10,5)
    plt.subplot(1, 2, 1)
    plt.plot(np.mean(loss['train'], axis=1))
    plt.plot(np.mean(loss['test'], axis=1))
    plt.title('Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(('Training','Test'))
    plt.subplot(1, 2, 2)
    plt.plot(np.mean(acc['train'],axis=1))
    plt.plot(np.mean(acc['test'],axis=1))
    plt.title('Accuracy',fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('p(correct)')
    plt.legend(('Training','Test'))

def plot_mds(xyz,numbers, ctx):
    if len(ctx):
        n_samples = xyz.shape[0]
        plt.plot(xyz[0:n_samples//2,0], xyz[0:n_samples//2,1],'o',color='orange')
        plt.plot(xyz[n_samples//2:,0], xyz[n_samples//2:,1],'o',color='black')
    else:
        plt.plot(xyz[:,0],xyz[:,1],'.',color='white')
    for ii in range(0,xyz.shape[0]):
        if numbers[ii]%2==0 and numbers[ii]>4:
            plt.text(xyz[ii,0],xyz[ii,1],str(numbers[ii]),color=(0, 0,1),fontsize=14)
        elif numbers[ii]%2==0 and numbers[ii]<5:
            plt.text(xyz[ii,0],xyz[ii,1],str(numbers[ii]),color=(0, 0,.5),fontsize=14)
        elif numbers[ii]%2!=0 and numbers[ii]>4:
            plt.text(xyz[ii,0],xyz[ii,1],str(numbers[ii]),color=(1, 0, 0),fontsize=14)
        elif numbers[ii]%2!=0 and numbers[ii]<5:
            plt.text(xyz[ii,0],xyz[ii,1],str(numbers[ii]),color=(.5, 0, 0),fontsize=14)



def plot_mds3(xyz, numbers, ctx):
    """
    plots 3d scatter of mds projections,
    to visualise variation in context, magnitude and parity
    """
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10,10)
    ax = fig.gca(projection='3d')

    n_samples = xyz.shape[0]
    ax.plot(xyz[0:n_samples//2,0], xyz[0:n_samples//2,1], xyz[0:n_samples//2,2],'o',color='orange')
    ax.plot(xyz[n_samples//2:,0], xyz[n_samples//2:,1], xyz[n_samples//2:,2],'o',color='black')
    for ii in range(0,xyz.shape[0]):
        if numbers[ii]%2==0 and numbers[ii]>4:
            ax.text(xyz[ii,0],xyz[ii,1],xyz[ii,2],str(numbers[ii]),color=(0, 0,1),fontsize=16)
        elif numbers[ii]%2==0 and numbers[ii]<5:
            ax.text(xyz[ii,0],xyz[ii,1],xyz[ii,2],str(numbers[ii]),color=(0, 0,.5),fontsize=16)
        elif numbers[ii]%2!=0 and numbers[ii]>4:
            ax.text(xyz[ii,0],xyz[ii,1],xyz[ii,2],str(numbers[ii]),color=(1, 0, 0),fontsize=16)
        elif numbers[ii]%2!=0 and numbers[ii]<5:
            ax.text(xyz[ii,0],xyz[ii,1],xyz[ii,2],str(numbers[ii]),color=(.5, 0, 0),fontsize=16)
#     ax.axis('equal')

def subplot_mds(xyzs, numbers, figsize=(20, 10), ndims=2,ctx=[]):
    plt.figure()
    plt.rcParams["figure.figsize"] = figsize
    n_mds = len(xyzs)
    for ii, label in enumerate(xyzs.keys()):
        plt.subplot(round(n_mds/2)+1, 2, ii+1)
        if ndims==2:
            plot_mds(xyzs[label], numbers,ctx)
        elif ndims==3:
            plot_mds3(xyz[label], numbers, ctx)
        plt.title(label, fontweight='bold', fontsize=24)

def plot_rdm(rdm, hasctx=0):
    rdm = (rdm-np.min(rdm))/(np.max(rdm)-np.min(rdm))
    plt.imshow(rdm)
    if hasctx:
        labels = ['c' + str(ii) + 'n' + str(jj) for ii in range(1,3) for jj in range(1,9)]
        plt.xticks(np.arange(0,16),labels,fontweight='bold', fontsize=14, rotation='vertical')
        plt.yticks(np.arange(0,16),labels, fontweight='bold', fontsize=14)
    else:
        labels = ['n' + str(jj) for jj in range(1,9)]
        plt.xticks(np.arange(0,8),labels,fontweight='bold', fontsize=14)
        plt.yticks(np.arange(0,8),labels, fontweight='bold', fontsize=14)

#     plt.colorbar(label='dissimilarity')

def subplot_rdms(rdms,figsize=(20, 10),hasctx=0):
    plt.figure()
    plt.rcParams["figure.figsize"] = figsize
    n_rdms = len(rdms)
    for ii,label in enumerate(rdms.keys()):
        plt.subplot(round(n_rdms/2)+1,2,ii+1)
        plot_rdm(rdms[label],hasctx)
        plt.title(label, fontweight='bold', fontsize=24)
