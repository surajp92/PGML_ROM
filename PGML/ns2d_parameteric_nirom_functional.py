# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:41:10 2019

@author: Suraj
"""
#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import simps
import pyfftw

from numpy.random import seed
seed(10)
from tensorflow import set_random_seed
set_random_seed(10)
import pandas as pd
import time as clck
import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import load_model

import keras.backend as K
K.set_floatx('float64')

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


#%% Define Functions

###############################################################################
#POD Routines
###############################################################################         
def POD(u,R): #Basis Construction
    n,ns = u.shape
    U,S,Vh = LA.svd(u, full_matrices=False)
    Phi = U[:,:R]  
    L = S**2
    #compute RIC (relative inportance index)
    RIC = sum(L[:R])/sum(L)*100   
    return Phi,L,RIC

def PODproj(u,Phi): #Projection
    a = np.dot(u.T,Phi)  # u = Phi * a.T
    return a

def PODrec(a,Phi): #Reconstruction    
    u = np.dot(Phi,a.T)    
    return u


###############################################################################
#Interpolation Routines
###############################################################################  
# Grassmann Interpolation
def GrassInt(Phi,pref,p,pTest):
    # Phi is input basis [training]
    # pref is the reference basis [arbitrarty] for Grassmann interpolation
    # p is the set of training parameters
    # pTest is the testing parameter
    
    nx,nr,nc = Phi.shape
    Phi0 = Phi[:,:,pref] 
    Phi0H = Phi0.T 

    print('Calculating Gammas...')
    Gamma = np.zeros((nx,nr,nc))
    for i in range(nc):
        templ = Phi[:,:,i] - LA.multi_dot([Phi0,Phi0H,Phi[:,:,i]])
        tempr = LA.inv( np.dot(Phi0H,Phi[:,:,i]) )
        temp = np.dot(templ, tempr)
                       
        U, S, Vh = LA.svd(temp, full_matrices=False)
        S = np.diag(S)
        Gamma[:,:,i] = LA.multi_dot([U,np.arctan(S),Vh])
    
    print('Interpolating ...')
    alpha = np.ones(nc)
    GammaL = np.zeros((nx,nr))
    #% Lagrange Interpolation
    for i in range(nc):
        for j in range(nc):
            if (j != i) :
                alpha[i] = alpha[i]*(pTest-p[j])/(p[i]-p[j])
    for i in range(nc):
        GammaL = GammaL + alpha[i] * Gamma[:,:,i]
            
    U, S, Vh = LA.svd(GammaL, full_matrices=False)
    PhiL = LA.multi_dot([ Phi0 , Vh.T ,np.diag(np.cos(S)) ]) + \
           LA.multi_dot([ U , np.diag(np.sin(S)) ])
    PhiL = PhiL.dot(Vh)
    return PhiL

###############################################################################
#LSTM Routines
############################################################################### 
def create_training_data_lstm(training_set, m, n, lookback):
    ytrain = training_set[lookback:,:]
    
    xtrain = np.zeros((m-lookback,lookback,n))
    
    for i in range(m-lookback):
        for j in range(lookback):
            xtrain[i,j,:] = training_set[i+j,:]
    return xtrain , ytrain

def create_training_data_lstm_re(training_set, m, n, re, lookback):
    ytrain = training_set[lookback:,:]
    
    xtrain = np.zeros((m-lookback,lookback,n+1))
    
    xtrain[:,:,0] = re
    
    for i in range(m-lookback):
        for j in range(lookback):
            xtrain[i,j,1:] = training_set[i+j,:]
    
    return xtrain , ytrain

###############################################################################
#Plotting Routines
############################################################################### 
def plot_3d_surface(x,t,field):
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(t, x)
    
    surf = ax.plot_surface(Y, X, field, cmap=plt.cm.viridis,
                           linewidth=1, antialiased=False,rstride=1,
                            cstride=1)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    fig.tight_layout()
    plt.show()
    fig.savefig('3d.png', dpi=200)

def plot_data_basis(x,y,PHI,filename):
    fig, ax = plt.subplots(nrows=int(nr/4),ncols=4,figsize=(13,6))
    ax = ax.flat
    nrs = at.shape[1]
    
    for i in range(nrs):
        f = np.zeros((nx+1,ny+1))
        f[:nx,:ny] = np.reshape(PHI[:,i],[nx,ny])
        f[:,ny] = f[:,1]
        f[nx,:] = f[1,:]
        
        cs = ax[i].contourf(x,y,f.T, 10, cmap = 'jet')
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(cs,cax=cax,orientation='vertical')
        ax[i].set_aspect(1.0)
        ax[i].set_title(r'$\phi_{'+str(i+1) + '}$')
        
    fig.tight_layout()    
    plt.show()
    fig.savefig(filename)

def plot_final_field(x,y,w_fom, w_true, w_gp, w_ml,filename):
    m = [w_fom, w_true, w_gp, w_ml]
    title = ['FOM','True','GP','LSTM']
    
    k = 0
    
    fig, axes = plt.subplots(nrows=1, ncols=4,figsize=(12,3))
    
    ax = axes.flat
    
    for i in range(4):
        cs = ax[i].contour(x,y,m[i].T, 20, cmap = 'coolwarm', vmin=0, vmax=1.0)
        ax[i].set_aspect(1.0)
        if k<4:
            ax[i].set_title(title[k],fontsize='14')
        ax[i].set_xticks([0,2,4,6])
        ax[i].set_yticks([0,2,4,6])
        k = k+1
        
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    cbar_ax = fig.add_axes([0.16, -0.1, 0.7, 0.05])
    cbar = fig.colorbar(cs, cax=cbar_ax,orientation='horizontal')
    plt.show()
    fig.savefig(filename,dpi=200,bbox_inches='tight')
    #cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in [-0.2,0,0.2,0.4,0.6,0.8,1.0]])
    
def plot_true_gp(t,at,aGP,filename):
    fig, ax = plt.subplots(nrows=int(nr/2),ncols=2,figsize=(10,8))
    ax = ax.flat
    nrs = at.shape[1]
    
    for i in range(nrs):
        ax[i].plot(t,at[:,i],'k',label=r'True Values')
        ax[i].plot(t,aGP[:,i],'b--',label=r'Exact Values')
        ax[i].set_xlabel(r'$t$',fontsize=14)
        ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
        ax[i].set_xlim([min(t), max(t)])
#        ax[i].set_ylim([1.2*np.min(aGP[:,i]),1.2*np.max(aGP[:,i])])
        ax[i].set_xticks(np.arange(min(t), max(t)+0.5, 5))
    
    fig.tight_layout()
    
    fig.subplots_adjust(bottom=0.1)
    line_labels = ["True","Standard GP"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
    plt.show()
    fig.savefig(filename)

def plot_true_gp_lstm(t,aTrue,aGPtest,aML,filename):
    fig, ax = plt.subplots(nrows=int(nr/2),ncols=2,figsize=(10,8))
    ax = ax.flat
    nrs = aTrue.shape[1]
    
    for i in range(nrs):
        ax[i].plot(t,aTrue[:,i],'k',label=r'True Values')
        ax[i].plot(t,aGPtest[:,i],'b--',label=r'Exact Values')
        ax[i].plot(t,aML[:,i],'m-.',label=r'Exact Values')
        ax[i].set_xlabel(r'$t$',fontsize=14)
        ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
        ax[i].set_xlim([min(t), max(t)])
#        ax[i].set_ylim([1.2*np.min(aGPtest[:,i]),1.2*np.max(aGPtest[:,i])])
        ax[i].set_xticks(np.arange(min(t), max(t)+0.5, 5))
        
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    
    line_labels = ["True","GP","LSTM"]#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=4, labelspacing=0.)
    plt.show()
    fig.savefig(filename, dpi=200)

def plot_true_gp_lstm_uq(t,aTrue,aGPtest,aML,aML_min,aML_max,filename):
    fig, ax = plt.subplots(nrows=int(nr/2),ncols=2,figsize=(10,8))
    ax = ax.flat
    nrs = aTrue.shape[1]
    
    for i in range(nrs):
        ax[i].plot(t,aTrue[:,i],'k')
        ax[i].plot(t,aGPtest[:,i],'b')
        ax[i].plot(t,aML[:,i],'m')
        ax[i].fill_between(t, aML_min[:,i], aML_max[:,i], color='darkorange', alpha=0.3)
        
        ax[i].set_xlabel(r'$t$',fontsize=14)
        ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
        ax[i].set_xlim([min(t), max(t)])
#        ax[i].set_ylim([1.2*np.min(aGPtest[:,i]),1.2*np.max(aGPtest[:,i])])
        ax[i].set_xticks(np.arange(min(t), max(t)+0.5, 5))
        
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    
    line_labels = ["True","GP","LSTM"]#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=4, labelspacing=0.)
    plt.show()
    fig.savefig(filename, dpi=200)

def plot_true_gp_lstm_uq2(t,aTrue,aGPtest,aML,aML_min,aML_max,aML_all,filename):
    fig, ax = plt.subplots(nrows=int(nr/2),ncols=2,figsize=(10,8))
    ax = ax.flat
    nrs = aTrue.shape[1]
    
    for i in range(nrs):
        ax[i].plot(t,aTrue[:,i],'k')
        ax[i].plot(t,aGPtest[:,i],'b')
        for j in range(num_ensembles):
            ax[i].plot(t,aML_all[j,:,i],'m', alpha=0.1)
        ax[i].plot(t,aML[:,i],'m')
#        ax[i].fill_between(t, aML_min[:,i], aML_max[:,i], color='darkorange', alpha=0.8)
        
        ax[i].set_xlabel(r'$t$',fontsize=14)
        ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
        ax[i].set_xlim([min(t), max(t)])
#        ax[i].set_ylim([1.2*np.min(aGPtest[:,i]),1.2*np.max(aGPtest[:,i])])
        ax[i].set_xticks(np.arange(min(t), max(t)+0.5, 5))
        
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    
    line_labels = ["True","GP","LSTM"]#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=4, labelspacing=0.)
    plt.show()
    fig.savefig(filename, dpi=200)   

def plot_data_allmodes_train(t,at,filename):
    fig, ax = plt.subplots(nrows=int(nr/2),ncols=2,figsize=(12,8))
    ax = ax.flat
    nrs = at.shape[1]
    
    for i in range(nrs):
        ax[i].plot(t,at[:,i],label=str(i+1))
        ax[i].set_xlabel(r'$t$',fontsize=14)
        ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)   
        ax[i].set_xlim([min(t), max(t)])
#        ax[i].set_ylim([1.2*np.min(aGPtest[:,i]),1.2*np.max(aGPtest[:,i])])
        ax[i].set_xticks(np.arange(min(t), max(t)+0.5, 5))
    
    fig.tight_layout()
    
    fig.subplots_adjust(bottom=0.1)
    line_labels = ["Re=200","Re=400","Re=600","Re=800"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=4, labelspacing=0.)
    plt.show()
    fig.savefig(filename, dpi=200)
    
def plot_data_allmodes(t,at,aTest,aLSTM,filename):
    fig, ax = plt.subplots(nrows=int(nr/2),ncols=2,figsize=(12,8))
    ax = ax.flat
    nrs = at.shape[1]
    
    for i in range(nrs):
        #for k in range(at.shape[2]):
        ax[i].plot(t,at[:,i],label=str(k))
        ax[i].plot(t,aTest[:,i],'k--',label=str(k))
        ax[i].plot(t,aLSTM[:,i],'b-.',label=str(k))
        #ax[i].legend(loc=0)
        #ax[i].plot(t,aGP[:,i],label=r'Exact Values')
        #ax[i].plot(t,aGPm[:,i],'r-.',label=r'True Values')
        ax[i].set_xlabel(r'$t$',fontsize=14)
        ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)   
        ax[i].set_xlim([min(t), max(t)])
#        ax[i].set_ylim([1.2*np.min(aGPtest[:,i]),1.2*np.max(aGPtest[:,i])])
        ax[i].set_xticks(np.arange(min(t), max(t)+0.5, 5))
    
    fig.tight_layout()
    
    fig.subplots_adjust(bottom=0.1)
    line_labels = ["Re=200","Re=400","Re=600","Re=800","Test="+str(ReTest),"LSTM"]#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=6, labelspacing=0.)
    plt.show()
    fig.savefig(filename, dpi=200)
    
###############################################################################
#GP Routines
###############################################################################     
def rhs(nr, b_l, b_nl, a): # Right Handside of Galerkin Projection
    r2, r3, r = [np.zeros(nr) for _ in range(3)]
    
    for k in range(nr):
        r2[k] = 0.0
        for i in range(nr):
            r2[k] = r2[k] + b_l[i,k]*a[i]
    
    for k in range(nr):
        r3[k] = 0.0
        for j in range(nr):
            for i in range(nr):
                r3[k] = r3[k] + b_nl[i,j,k]*a[i]*a[j]
    
    r = r2 + r3    
    return r

# fast poisson solver using second-order central difference scheme
def fpsi(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    hx = 2.0*np.pi/np.float64(nx)
    hy = 2.0*np.pi/np.float64(ny)
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[:] = hx*np.float64(np.arange(0, nx))

    ky[:] = hy*np.float64(np.arange(0, ny))
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(np.cos(kx), np.cos(ky), indexing='ij')
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f,0.0)

    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    e = fft_object(data)
    #e = pyfftw.interfaces.scipy_fftpack.fft2(data)
    
    e[0,0] = 0.0
    
    data1[:,:] = e[:,:]/(aa + bb*kx[:,:] + cc*ky[:,:])

    ut = np.real(fft_object_inv(data1))
        
    return ut

def nonlinear_term(nx,ny,dx,dy,wf,sf):
    '''
    this function returns -(Jacobian)
    
    '''
    w = np.zeros((nx+3,ny+3))
    
    w[1:nx+1,1:ny+1] = wf
    
    # periodic
    w[:,ny+1] = w[:,1]
    w[nx+1,:] = w[1,:]
    w[nx+1,ny+1] = w[1,1]
    
    # ghost points
    w[:,0] = w[:,ny]
    w[:,ny+2] = w[:,2]
    w[0,:] = w[nx,:]
    w[nx+2,:] = w[2,:]
    
    s = np.zeros((nx+3,ny+3))
    
    s[1:nx+1,1:ny+1] = sf
    
    # periodic
    s[:,ny+1] = s[:,1]
    s[nx+1,:] = s[1,:]
    s[nx+1,ny+1] = s[1,1]
    
    # ghost points
    s[:,0] = s[:,ny]
    s[:,ny+2] = s[:,2]
    s[0,:] = s[nx,:]
    s[nx+2,:] = s[2,:]
    
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    f = np.zeros((nx+1,ny+1))
    
    #Arakawa
    j1 = gg*( (w[2:nx+3,1:ny+2]-w[0:nx+1,1:ny+2])*(s[1:nx+2,2:ny+3]-s[1:nx+2,0:ny+1]) \
             -(w[1:nx+2,2:ny+3]-w[1:nx+2,0:ny+1])*(s[2:nx+3,1:ny+2]-s[0:nx+1,1:ny+2]))

    j2 = gg*( w[2:nx+3,1:ny+2]*(s[2:nx+3,2:ny+3]-s[2:nx+3,0:ny+1]) \
            - w[0:nx+1,1:ny+2]*(s[0:nx+1,2:ny+3]-s[0:nx+1,0:ny+1]) \
            - w[1:nx+2,2:ny+3]*(s[2:nx+3,2:ny+3]-s[0:nx+1,2:ny+3]) \
            + w[1:nx+2,0:ny+1]*(s[2:nx+3,0:ny+1]-s[0:nx+1,0:ny+1]))

    j3 = gg*( w[2:nx+3,2:ny+3]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[0:nx+1,0:ny+1]*(s[0:nx+1,1:ny+2]-s[1:nx+2,0:ny+1]) \
            - w[0:nx+1,2:ny+3]*(s[1:nx+2,2:ny+3]-s[0:nx+1,1:ny+2]) \
            + w[2:nx+3,0:ny+1]*(s[2:nx+3,1:ny+2]-s[1:nx+2,0:ny+1]) )

    f = -(j1+j2+j3)*hh
                  
    return f[1:nx+1,1:ny+1]

def linear_term(nx,ny,dx,dy,re,f):
    w = np.zeros((nx+3,ny+3))
    
    w[1:nx+1,1:ny+1] = f
    
    # periodic
    w[:,ny+1] = w[:,1]
    w[nx+1,:] = w[1,:]
    w[nx+1,ny+1] = w[1,1]
    
    # ghost points
    w[:,0] = w[:,ny]
    w[:,ny+2] = w[:,2]
    w[0,:] = w[nx,:]
    w[nx+2,:] = w[2,:]
    
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    
    f = np.zeros((nx+1,ny+1))
    
    lap = aa*(w[2:nx+3,1:ny+2]-2.0*w[1:nx+2,1:ny+2]+w[0:nx+1,1:ny+2]) \
        + bb*(w[1:nx+2,2:ny+3]-2.0*w[1:nx+2,1:ny+2]+w[1:nx+2,0:ny+1])
    
    f = lap/re
            
    return f[1:nx+1,1:ny+1]

def pbc(w):
    f = np.zeros((nx+1,ny+1))
    f[:nx,:ny] = w
    f[:,ny] = f[:,0]
    f[nx,:] = f[0,:]
    
    return f

#%% Main program:
# Inputs
nx =  256  #spatial grid number
ny = 256
nc = 4     #number of control parameters (nu)
ns = 200    #number of snapshot per each Parameter 
nr = 8      #number of modes
Re_start = 200.0
Re_final = 800.0
Re  = np.linspace(Re_start, Re_final, nc) #control Reynolds
nu = 1/Re   #control dissipation
lx = 2.0*np.pi
ly = 2.0*np.pi
dx = lx/nx
dy = ly/ny
dt = 1e-1
tm = 20.0

ReTest = 1000.0
pref = 3 #Reference case in [0:nRe]

num_ensembles = 30

Training = False

#%% Data generation for training
x = np.linspace(0, lx, nx+1)
y = np.linspace(0, ly, ny+1)
t = np.linspace(0, tm, ns+1)

um = np.zeros(((nx)*(ny), ns+1, nc))
up = np.zeros(((nx)*(ny), ns+1, nc))
uo = np.zeros(((nx)*(ny), ns+1, nc))

for p in range(0,nc):
    for n in range(0,ns+1):
        file_input = "../../snapshots/Re_"+str(int(Re[p]))+"/w/w_"+str(int(n))+ ".npy"
        w = np.load(file_input)
        #w = np.genfromtxt(file_input, delimiter=',')
        
        w1 = w[1:nx+1,1:ny+1]
        
        um[:,n,p] = np.reshape(w1,(nx)*(ny)) #snapshots from unperturbed solution
        uo[:,n,p] = um[:,n,p] 

#%% POD basis computation
PHIw = np.zeros(((nx)*(ny),nr,nc))
PHIs = np.zeros(((nx)*(ny),nr,nc))        
       
L = np.zeros((ns+1,nc)) #Eigenvalues      
RIC = np.zeros((nc))    #Relative information content

print('Computing POD basis for vorticity ...')
for p in range(0,nc):
    u = uo[:,:,p]
    PHIw[:,:,p], L[:,p], RIC[p]  = POD(u, nr) 

#%% Calculating true POD coefficients (observed)
at = np.zeros((ns+1,nr,nc))
print('Computing true POD coefficients...')
for p in range(nc):
    at[:,:,p] = PODproj(uo[:,:,p],PHIw[:,:,p])

at_signs = np.sign(at[0,:,:])
at = at/at_signs
PHIw = PHIw/(at_signs)

plot_data_allmodes_train(t,at,'all_modes_train.png')

#%%    
print('Computing POD basis for streamfunction ...')
for p in range(0,nc):
    for i in range(nr):
        phi_w = np.reshape(PHIw[:,i,p],[nx,ny])
        phi_s = fpsi(nx, ny, dx, dy, -phi_w)
        PHIs[:,i,p] = np.reshape(phi_s,(nx)*(ny))
        
#%%
at_modes = np.zeros((nc,ns+1,nr))
phi_basis = np.zeros((nc,nx*ny,nr))

for i in range(nc):
    at_modes[i,:,:] = at[:,:,i]
    phi_basis[i,:,:] = PHIw[:,:,i]

with open("./plotting/all_modes.csv", 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(at_modes.shape))
    for data_slice in at_modes:
        np.savetxt(outfile, data_slice, delimiter=",")
        outfile.write('# New slice\n')

with open("./plotting/all_basis.csv", 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(phi_basis.shape))
    for data_slice in phi_basis:
        np.savetxt(outfile, data_slice, delimiter=",")
        outfile.write('# New slice\n')
                      
#%%
print('Reconstructing with true coefficients')
w = PODrec(at[:,:,0],PHIw[:,:,0])

w = w[:,-1]
w = np.reshape(w,[nx,ny])

wt = np.reshape(um[:,-1,0],[nx,ny])

fig, axs = plt.subplots(1,2,sharey=True,figsize=(8,4))
cs = axs[0].contourf(x[:nx],y[:ny],w.T, 120, cmap = 'jet')
axs[0].set_aspect(1.0)

cs = axs[1].contourf(x[:nx],y[:ny],wt.T, 120, cmap = 'jet')
axs[1].set_aspect(1.0)

#fig.colorbar(cs,orientation='vertical')
fig.tight_layout() 
plt.show()
#fig.savefig("reconstructed.eps", bbox_inches = 'tight')

#%% Galerkin projection [Fully Intrusive]

###############################################################################
# Galerkin projection with nr
###############################################################################

b_l = np.zeros((nr,nr,nc))
b_nl = np.zeros((nr,nr,nr,nc))
linear_phi = np.zeros(((nx)*(ny),nr,nc))
nonlinear_phi = np.zeros(((nx)*(ny),nr,nc))

# linear term   
for p in range(nc):
    for i in range(nr):
        phi_w = np.reshape(PHIw[:,i,p],[nx,ny])
        
        lin_term = linear_term(nx,ny,dx,dy,Re[p],phi_w)
        linear_phi[:,i,p] = np.reshape(lin_term,(nx)*(ny))

for p in range(nc):
    for k in range(nr):
        for i in range(nr):
            b_l[i,k,p] = np.dot(linear_phi[:,i,p].T , PHIw[:,k,p]) 
                   
# nonlinear term 
for p in range(nc):
    for i in range(nr):
        phi_w = np.reshape(PHIw[:,i,p],[nx,ny])
        for j in range(nr):  
            phi_s = np.reshape(PHIs[:,j,p],[nx,ny])
            nonlin_term = nonlinear_term(nx,ny,dx,dy,phi_w,phi_s)
            jacobian_phi = np.reshape(nonlin_term,(nx)*(ny))
            for k in range(nr):    
                b_nl[i,j,k,p] = np.dot(jacobian_phi.T, PHIw[:,k,p]) 

#%% solving ROM by Adams-Bashforth scheme          
aGP = np.zeros((ns+1,nr,nc))
for p in range(nc):
    aGP[0,:,p] = at[0,:nr,p]
    aGP[1,:,p] = at[1,:nr,p]
    aGP[2,:,p] = at[2,:nr,p]
    for k in range(3,ns+1):
        r1 = rhs(nr, b_l[:,:,p], b_nl[:,:,:,p], aGP[k-1,:,p])
        r2 = rhs(nr, b_l[:,:,p], b_nl[:,:,:,p], aGP[k-2,:,p])
        r3 = rhs(nr, b_l[:,:,p], b_nl[:,:,:,p], aGP[k-3,:,p])
        temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
        aGP[k,:,p] = aGP[k-1,:,p] + dt*temp 
        
#%%
plot_true_gp(t,at[:,:,-1],aGP[:,:,-1],f'modes_{Re[-1]}.png')    
plot_data_basis(x,y,PHIw[:,:,-1], f'basis_{Re[-1]}.png')

#%%
lookback = 3 #Number of lookbacks

# use xtrain from here
for p in range(nc):
    xt, yt = create_training_data_lstm_re(at[:,:,p], ns+1, nr, Re[p], lookback)
    xt_gp, yt_gp = create_training_data_lstm_re(aGP[:,:,p], ns+1, nr, Re[p], lookback)
    if p == 0:
        xtrain = xt
        xtrain_gp = xt_gp[:,:,1:]
        ytrain = yt
    else:
        xtrain = np.vstack((xtrain,xt))
        xtrain_gp = np.vstack((xtrain_gp,xt_gp[:,:,1:]))
        ytrain = np.vstack((ytrain,yt))

#%%
data = xtrain # modified GP as the input data
data_gp = xtrain_gp
labels = ytrain
        
#%%
# Scaling data
p,q,r = data.shape
data2d = data.reshape(p*q,r)

scalerIn = MinMaxScaler(feature_range=(-1,1))
scalerIn = scalerIn.fit(data2d)
data2d = scalerIn.transform(data2d)
data = data2d.reshape(p,q,r)

p_gp,q_gp,r_gp = data_gp.shape
data_gp2d = data_gp.reshape(p_gp*q_gp,r_gp)

scalerGP = MinMaxScaler(feature_range=(-1,1))
scalerGP = scalerGP.fit(data_gp2d)
data_gp2d = scalerGP.transform(data_gp2d)
data_gp = data_gp2d.reshape(p_gp,q_gp,r_gp)

scalerOut = MinMaxScaler(feature_range=(-1,1))
scalerOut = scalerOut.fit(labels)
labels = scalerOut.transform(labels)

#xtrain = np.copy(data)
#ytrain = np.copy(labels)

#%%
slice_list = np.arange(0,p,1)
slice_train, slice_valid, ytrain, yvalid = train_test_split(slice_list, labels, test_size=0.2 , shuffle= True)

xtrain = data[slice_train]
xvalid = data[slice_valid]

xtrain_gp = data_gp[slice_train]
xvalid_gp = data_gp[slice_valid]

#%%
ncells = 80
m,n = ytrain.shape
xi= Input(shape=(lookback, n+1))
xgp= Input(shape=(lookback,n))
xl = LSTM(ncells, return_sequences=True, activation='tanh')(xi)
xl = LSTM(ncells,  return_sequences=True, activation='tanh')(xl)
xl = LSTM(ncells,  return_sequences=True, activation='tanh')(xl)
xl = concatenate(inputs=[xl, xgp])
xl = LSTM(ncells, activation='tanh')(xl)
output = Dense(n)(xl)

model = Model(inputs=[xi,xgp], outputs=output)
print(model.summary())
tf.keras.utils.plot_model(model, to_file=f'pgml_rom_{ReTest}_{Re[pref]}.png', show_shapes=True)

tf.reset_default_graph()
        
#%%

if Training:
    for i in range(1,num_ensembles+1):
        seed_number = int(i*10)
        print(seed_number)
        import random
        random.seed(seed_number)
        seed(seed_number)
        set_random_seed(seed_number)

        m,n = ytrain.shape # m is number of training samples, n is number of output features [i.e., n=nr]
        
        # functional API
        xi= Input(shape=(lookback, n+1))
        xgp= Input(shape=(lookback,n))
        xl = LSTM(ncells, return_sequences=True, activation='tanh')(xi)
        xl = LSTM(ncells,  return_sequences=True, activation='tanh')(xl)
        xl = LSTM(ncells,  return_sequences=True, activation='tanh')(xl)
        xl = concatenate(inputs=[xl, xgp])
        xl = LSTM(ncells, activation='tanh')(xl)
        output = Dense(n)(xl)
        
        model = Model(inputs=[xi,xgp], outputs=output)
#        print(model.summary())
        
        # compile model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
        
        print(model.summary())
        
        # fit the model
        history = model.fit(x=[xtrain, xtrain_gp], y=ytrain, epochs=600, batch_size=32, 
                            validation_data= ([xvalid,xvalid_gp],yvalid))
        
        # evaluate the model
        scores = model.evaluate(x=[xtrain, xtrain_gp], y=ytrain, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        plt.figure()
        epochs = range(1, len(loss) + 1)
        plt.semilogy(epochs, loss, 'b', label='Training loss')
        plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        filename = f'loss_{i*10}.png'
        plt.savefig(filename, dpi = 400)
        plt.show()
        
        # Save the model
        filename = f'nirom_model_{i*10}.hd5'
        model.save(filename)
        
        tf.reset_default_graph()
        
#%% Testing
# Data generation for testing
uTest = np.zeros((nx*ny, ns+1))
upTest = np.zeros((nx*ny, ns+1))
uoTest = np.zeros((nx*ny, ns+1))
nuTest = 1/ReTest

for n in range(ns+1):
   file_input = "../../snapshots/Re_"+str(int(ReTest))+"/w/w_"+str(int(n))+ ".npy"
   w = np.load(file_input)
   #w = np.genfromtxt(file_input, delimiter=',')
    
   w1 = w[1:nx+1,1:ny+1]
    
   uTest[:,n] = np.reshape(w1,(nx)*(ny)) #snapshots from unperturbed solution
   uoTest[:,n] = uTest[:,n] 

#%%   
w_fom = uoTest[:,-1] # last time step
w_fom = np.reshape(w_fom,[nx,ny])
w_fom = pbc(w_fom)

#% POD basis computation     
print('Computing testing POD basis...')
PHItrue, Ltrue, RICtrue  = POD(uoTest, nr) 

#% Calculating true POD coefficients
print('Computing testing POD coefficients...')
aTrue = PODproj(uoTest,PHItrue)

aTrue_sign = np.sign(aTrue[0,:])
aTrue = aTrue/aTrue_sign
PHItrue = PHItrue/aTrue_sign

#%% Basis Interpolation
PHIwtest = GrassInt(PHIw,pref,nu,nuTest)

aTest = PODproj(uoTest,PHIwtest)

aTest_sign = np.sign(aTest[0,:])
aTest = aTest/aTest_sign
PHIwtest = PHIwtest/aTest_sign

print('Reconstructing with true coefficients for test Re')
w_test = PODrec(aTest[:,:],PHIwtest[:,:])

bases_test = np.zeros((2,nx*ny,nr))
bases_test[0] = PHItrue
bases_test[1] = PHIwtest

with open(f"./plotting/bases_{ReTest}_{Re[pref]}.csv", 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(bases_test.shape))
    for data_slice in bases_test:
        np.savetxt(outfile, data_slice, delimiter=",")
        outfile.write('# New slice\n')

#%%                      
w_test = w_test[:,-1]
w_test = np.reshape(w_test,[nx,ny])
w_test = pbc(w_test)

PHIstest = np.zeros(((nx)*(ny),nr))

for i in range(nr):
    phi_w = np.reshape(PHIwtest[:,i],[nx,ny])
    phi_s = fpsi(nx, ny, dx, dy, -phi_w)    
    PHIstest[:,i] = np.reshape(phi_s,(nx)*(ny))

#%%
plot_data_basis(x,y,PHItrue[:,:], f'basis_true_{ReTest}_{Re[pref]}.png')
plot_data_basis(x,y,PHIwtest[:,:], f'basis_grassman_{ReTest}_{Re[pref]}.png')

#%%
b_l = np.zeros((nr,nr))
b_nl = np.zeros((nr,nr,nr))
linear_phi = np.zeros(((nx)*(ny),nr))
nonlinear_phi = np.zeros(((nx)*(ny),nr))
 
for k in range(nr):
    phi_w = np.reshape(PHIwtest[:,k],[nx,ny])
    lin_term = linear_term(nx,ny,dx,dy,ReTest,phi_w)
    linear_phi[:,k] = np.reshape(lin_term,(nx)*(ny))

for k in range(nr):
    for i in range(nr):
        b_l[i,k] = np.dot(linear_phi[:,i].T , PHIwtest[:,k]) 
                   
for i in range(nr):
    phi_w = np.reshape(PHIwtest[:,i],[nx,ny])
    for j in range(nr):  
        phi_s = np.reshape(PHIstest[:,j],[nx,ny])
        nonlin_term = nonlinear_term(nx,ny,dx,dy,phi_w,phi_s)
        jacobian_phi = np.reshape(nonlin_term,(nx)*(ny))
        for k in range(nr):    
            b_nl[i,j,k] = np.dot(jacobian_phi.T, PHIwtest[:,k]) 
       
aGPtest = np.zeros((ns+1,nr))
aGPtest[0,:] = aTest[0,:nr]
aGPtest[1,:] = aTest[1,:nr]
aGPtest[2,:] = aTest[2,:nr]
for k in range(3,ns+1):
    r1 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPtest[k-1,:])
    r2 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPtest[k-2,:])
    r3 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPtest[k-3,:])
    temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
    aGPtest[k,:] = aGPtest[k-1,:] + dt*temp 

print('Reconstructing with GP coefficients for test Re')
w_gp = PODrec(aGPtest[:,:],PHIwtest[:,:])

w_gp = w_gp[:,-1]
w_gp = np.reshape(w_gp,[nx,ny])
w_gp = pbc(w_gp)
 
plot_true_gp(t,aTest[:,:],aGPtest[:,:],f'modes_test_{ReTest}_{Re[pref]}.png')

#%% LSTM [Fully Nonintrusive]
# testing
aML_all_seeds = np.zeros((num_ensembles,ns+1,nr))

for j in range(1,num_ensembles+1):
    print(j)
    model = load_model(f'nirom_model_{j*10}.hd5')
    model.get_config()
    
    testing_set = np.zeros((ns+1,nr+1))
    testing_set[:,0] = ReTest
    testing_set[:,1:] = aTest
    
    m,n = aTest.shape
    xtest = np.zeros((1,lookback,nr+1))
    xtest_gp = np.zeros((1,lookback,nr))
    
    aML = np.zeros((ns+1,nr))
    
    # Initializing
    for i in range(lookback):
        xtest[0,i,:] = testing_set[i,:].reshape(1,-1)
        xtest_gp[0,i,:] = aGPtest[i,:].reshape(1,-1)
        aML[i,:] =  testing_set[i,1:]
    
    # Prediction
    for i in range(lookback,ns+1):
        xtest_sc = scalerIn.transform(xtest[0])
        xtest_sc = xtest_sc.reshape(1,lookback,nr+1)
        xtest_gp_sc = scalerGP.transform(xtest_gp[0])
        xtest_gp_sc = xtest_gp_sc.reshape(1,lookback,nr)
        
        ytest_sc = model.predict([xtest_sc,xtest_gp_sc])
        ytest = scalerOut.inverse_transform(ytest_sc) # residual/ correction
        
        aML[i,:] =  ytest
        
        for k in range(lookback-1):
            xtest[0,k,1:] = xtest[0,k+1,1:]
            xtest_gp[0,k,:] = xtest_gp[0,k+1,:]
            
        xtest[0,lookback-1,1:] =  ytest
        xtest_gp[0,lookback-1,:] =  aGPtest[i,:]
    
    aML_all_seeds[j-1,:,:] = aML

#%%
aML_avg = np.average(aML_all_seeds, axis = 0)
aML_min = np.min(aML_all_seeds, axis = 0)
aML_max = np.max(aML_all_seeds, axis = 0)    
aML_std = np.std(aML_all_seeds, axis = 0)

#%%    

print('Reconstructing with true coefficients for test Re')
w_ml = PODrec(aML_avg[:,:],PHIwtest[:,:])

w_ml = w_ml[:,-1]
w_ml = np.reshape(w_ml,[nx,ny])
w_ml = pbc(w_ml)

#%%
modal_coeffs = np.hstack((aTest,aGPtest,aML_avg))
field = np.zeros((4,w_fom.shape[0],w_fom.shape[1]))
field[0] = w_fom
field[1] = w_test
field[2] = w_gp
field[3] = w_ml

filename = f"./plotting/modes_{ReTest}_{Re[pref]}.csv"
np.savetxt(filename, modal_coeffs, delimiter=",")
    
with open(f"./plotting/field_{ReTest}_{Re[pref]}.csv", 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(field.shape))
    for data_slice in field:
        np.savetxt(outfile, data_slice, delimiter=",")
        outfile.write('# New slice\n')

#%%
plot_true_gp_lstm(t,aTest,aGPtest,aML_avg,f'test_modes_{ReTest}_{Re[pref]}.png')
plot_true_gp_lstm_uq(t,aTest,aGPtest,aML_avg,aML_min,aML_max,f'test_modes_uq_{ReTest}_{Re[pref]}_v1.png')    
plot_true_gp_lstm_uq2(t,aTest,aGPtest,aML_avg,aML_min,aML_max,aML_all_seeds,f'test_modes_uq_{ReTest}_{Re[pref]}_v2.png')    
plot_true_gp_lstm_uq(t,aTest,aGPtest,aML_avg,aML_avg-aML_std,aML_avg+aML_std,f'test_modes_uq_{ReTest}_{Re[pref]}_v3.png')    

plot_final_field(x,y, w_fom, w_test, w_gp, w_ml, f'field_{ReTest}_{Re[pref]}.png')     
plot_data_allmodes(t,at[:,:],aTest,aML_avg,f'allmodes_{ReTest}_{Re[pref]}.png')#,aGP1[-1,:,:])

np.savez(f'./plotting/modal_coefficients_{ReTest}_{Re[pref]}.npz', 
         t = t, aTest = aTest, aGPtest = aGPtest, aML = aML_all_seeds)

np.savez(f'./plotting/vorticity_field_{ReTest}_{Re[pref]}.npz', 
         w_fom = w_fom, w_test = w_test, w_gp = w_gp,
         w_ml = w_ml)

#%%
k = np.linspace(1,ns+1,201)

L_per = np.zeros(L.shape)
Ltrue_per = np.zeros(L.shape[0])
for n in range(L.shape[0]):
    L_per[n,:] = np.sum(L[:n],axis=0,keepdims=True)/np.sum(L,axis=0,keepdims=True)
    Ltrue_per[n] = np.sum(Ltrue[:n],axis=0,keepdims=True)/np.sum(Ltrue,axis=0,keepdims=True)

eigen_history = np.hstack((L,Ltrue.reshape(-1,1),L_per,Ltrue_per.reshape(-1,1)))
filename = f"./plotting/eigen_hist_{ReTest}_{Re[pref]}.csv"
np.savetxt(filename, eigen_history, delimiter=",")
  
#%% 
fig, axs = plt.subplots(1, 1, figsize=(7,5))#, constrained_layout=True)

axs.loglog(k,L[:,0], lw = 2, marker="o", linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
axs.loglog(k,L[:,1], lw = 2, marker="o", linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
axs.loglog(k,L[:,2],  lw = 2, marker="o", linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
axs.loglog(k,L[:,3],  lw = 2, marker="o", linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)

#axs.loglog(k,L[:,0], color='purple', lw = 2, marker="o", linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
#axs.loglog(k,L[:,1], color='orangered', lw = 2, marker="o", linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
#axs.loglog(k,L[:,2], color='navy', lw = 2, marker="o", linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
#axs.loglog(k,L[:,3], color='darkgreen', lw = 2, marker="o", linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)

#axs.loglog(k,Ltrue1, color='gray', lw = 2, marker="o", linestyle='--', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)

#axs.grid(True)
axs.axvspan(0, 4, alpha=0.1, color='blue')

axs.set_xlim([k[0], 20])
axs.set_ylim([10e-6,10e6])
axs.set_ylabel('$\sigma_k$', fontsize = 16)
axs.set_xlabel('$k$', fontsize = 16)

fig.tight_layout()

fig.subplots_adjust(bottom=0.2)
line_labels = ['Re = 200','Re = 400','Re = 600','Re = 800']#, "ML-Train", "ML-Test"]
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.1, ncol=4, labelspacing=0.,  prop={'size': 14} ) #bbox_to_anchor=(0.5, 0.0), borderaxespad=0.1, 

plt.show()
fig.savefig('eigen_256.png', dpi=200)

#%%
w_test = PODrec(aTest[:,:],PHIwtest[:,:])
w_gp = PODrec(aGPtest[:,:],PHIwtest[:,:])
w_ml = PODrec(aML_avg[:,:],PHIwtest[:,:])

np.savez(f'./plotting/w_all_times_{ReTest}.npz',
         PHItrue=PHItrue, aTrue=aTrue, w_fom = uoTest,
         PHIwtest=PHIwtest, aTest=aTest, w_test=w_test,
         aGPtest=aGPtest, w_gp=w_gp,
         aML_avg=aML_avg, w_ml=w_ml,
         aML_all_seeds=aML_all_seeds
         )