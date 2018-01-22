#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:27:31 2017

@author: naamahadad
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import DisAdvNet

home = os.path.expanduser('~')

params = {}
data_type = 'sprites'
params['res_path'] = 'models'
params['BN'] = True
params['EncBN'] = False
params['DecBN'] = False
params['Init'] = 'glorot_normal'
params['Activation'] = 'relu'

params['recweight'] = 0
params['sclsweight'] = 0
params['swap1weight'] = 0
params['swap2weight'] = 0
params['klweightZ'] = 0

params['original_dim'] = 28
params['filter_size'] = 5
params['l_size'] = 256
params['s_dim'] = 16
params['l_size'] = 16
params['latent_dim'] = 16
params['dp'] = 0.5
params['s_clss'] = 10
params['mseloss'] = False
params['batch_size'] = 100

params['RGB'] = 3
params['n_features'] = 64
params['original_dim'] = 32
params['s_dim'] = 32
params['latent_dim'] = 128
params['s_clss'] = 318
  
params['mainnet'] ='040517_074241'

#data_root = '/media/data2/naamahadad/Pylearn2/data/Sprites/flp64/'#X60pix/'#rem21/flp64/'
#x_test = np.load(data_root+'X_test.npy')
#y_test = np.load(data_root+'Y_test.npy')
x_test = np.load('data/Sprites_X_test_partial.npy')
y_test = np.load('data/Sprites_Y_test_partial.npy')

print 'Xtest shape: ' ,x_test.shape
print 'ytest shape: ' ,y_test.shape

perm_inds = np.random.permutation(range(x_test.shape[0]))
maxSamples = (np.floor(x_test.shape[0]/(params['batch_size']))*params['batch_size']).astype(np.int64)
#maxSamples = np.int64(5*params['batch_size'])#.astype(np.int64)
x_test = x_test[perm_inds[:maxSamples],:,:,:]
y_test = y_test[perm_inds[:maxSamples]]

net = DisAdvNet.DisAdvNet(params,'','',False)

print 'loading net...'
net.Snet.load_weights(params['res_path']+'/'+params['mainnet']+'_'+data_type+'_weights.h5_Sfreeze_weights.h5')
net.EncZ.load_weights(params['res_path']+'/' +params['mainnet']+'_'+data_type+'_weights.h5_encZ_weights.h5')

net.AdvNet.load_weights(params['res_path']+'/' +params['mainnet']+'_'+data_type+'_weights.h5_advNet_weights.h5')
net.Dec.load_weights(params['res_path']+'/'+params['mainnet']+'_'+data_type+'_weights.h5_decS_weights.h5')
print 'net loaded'

z_m_test, z_std_test, s_test = net.predictEnc(x_test)
print 'zdist: ',np.mean(z_m_test),np.std(z_m_test)
print 'sdist: ',np.mean(s_test),np.std(s_test)

preds_s = net.Sclsfr.predict(s_test,batch_size=params['batch_size'])

#z_test = z_m_test + (np.exp(z_std_test / 2) * np.random.normal(0, 1, (z_m_test.shape)))
z_test = z_m_test
print 'z_test_shape: ' ,z_test.shape
print 's_test: ' ,s_test.shape

X_test_dec = net.Dec.predict([z_test,s_test],batch_size=params['batch_size'])
sdim = s_test.shape[1]

n=10
full_inds = np.zeros((n,))
y_vals = np.unique(y_test)
if len(y_vals)>n:
    y_vals = np.random.permutation(y_vals)
for i in range(n):
    dig = y_vals[i%len(y_vals)]
    #print dig
    cur_inds = np.flatnonzero(y_test==dig)
    perm_inds = np.random.permutation(range(len(cur_inds)))
    full_inds[i] = cur_inds[perm_inds[i]]

#print full_inds
full_inds = full_inds.astype(np.int)
z_rand = np.zeros((n*n,z_test.shape[1]))
#z_rand = z_test[:n*n,:]
s_rand = np.zeros((n*n,sdim))
for i in range(n):
    for jj in range(n):
        z_rand[i*n+jj,:] = z_test[full_inds[i],:]
        if sdim>1:
            s_rand[i*n+jj,:] = s_test[full_inds[jj],:]
        else:
            s_rand[i*n+jj] = s_test[full_inds[jj]]
  

X_recon = net.Dec.predict([z_rand,s_rand],batch_size=(n*n))

for i in range(n):
    # display original
    plt.figure(0, facecolor='white')
    plt.subplots_adjust(wspace = 0.001,hspace=0.001)
    ax = plt.subplot(n+1, n+1, i+2)
    if params['RGB']==3:
        plt.imshow(np.transpose(x_test[full_inds[i]].reshape(3, params['original_dim'],params['original_dim']),(1,2,0)))
    else:
        plt.imshow(x_test[full_inds[i]].reshape(params['original_dim'], params['original_dim']))
        plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(n+1, n+1, (n+1)*(i+1)+1)
    if params['RGB']==3:
        plt.imshow(np.transpose(x_test[full_inds[i]].reshape(3, params['original_dim'],params['original_dim']),(1,2,0)))
    else:
        plt.imshow(x_test[full_inds[i]].reshape(params['original_dim'], params['original_dim']))
        plt.gray()
        
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for jj in range(n):
        ax = plt.subplot(n+1, n+1, (n+1)*(i+1)+2+jj)
        if params['RGB']==3:
            plt.imshow(np.transpose(X_recon[i*n+jj].reshape(3, params['original_dim'],params['original_dim']),(1,2,0)))
        else:
            plt.imshow(X_recon[i*n+jj].reshape(params['original_dim'], params['original_dim']))
            plt.gray()
            
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()
    
step = 1.0/(n-1)
z_rand = np.zeros((n*n,z_test.shape[1]))
s_rand = np.zeros((n*n,sdim))
firstind = full_inds[1] #For Mnist 1&8 interpolation
secind = full_inds[8]

z1 = z_test[firstind,:]     
z2 = z_test[secind,:]
X1 = x_test[firstind,:,:,:]
X2 = x_test[secind,:,:,:]
if sdim>1:
    s1 = s_test[firstind,:]     
    s2 = s_test[secind,:]
else:
    s1 = s_test[firstind]     
    s2 = s_test[secind] 

for i in range(n):
    for jj in range(n):
        z_rand[i*n+jj,:] = z1*(1-step*i) + z2*step*i
        if sdim>1:
            s_rand[i*n+jj,:] = s1*(1-step*jj) + s2*step*jj
        else:
            s_rand[i*n+jj] = s1*(1-step*jj) + s2*step*jj
 
X_recon = net.Dec.predict([z_rand,s_rand],batch_size=(n*n))

plt.figure(1, facecolor='white')
plt.subplots_adjust(wspace = 0.001,hspace=0.001)
#fig2 = plt.figure()
for i in range(n):
    for jj in range(n):
        ax = plt.subplot(n, n, n*i+jj+1)
        #plt.imshow(X_recon[i*n+jj].reshape(28, 28))
        if params['RGB']==3:
            plt.imshow(np.transpose(X_recon[i*n+jj].reshape(3, params['original_dim'],params['original_dim']),(1,2,0)))
        else:
            if i==0 and jj==0:
                plt.imshow(X1.reshape(params['original_dim'], params['original_dim']))
            elif i==(n-1) and jj==(n-1):
                plt.imshow(X2.reshape(params['original_dim'], params['original_dim']))
            else:
                plt.imshow(X_recon[i*n+jj].reshape(params['original_dim'], params['original_dim']))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()
