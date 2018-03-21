from keras.datasets import mnist
import numpy as np
from time import strftime,localtime
import os
from keras.utils import np_utils
import DisAdvNet2
import matplotlib.pyplot as plt
home = os.path.expanduser('~')

params = {}
params['res_path'] = ''

params['BN'] = True
params['EncBN'] = False
params['DecBN'] = False
params['Init'] = 'glorot_normal'
params['Activation'] = 'relu'
params['l_size'] = 16

params['batch_size'] = 100
params['nb_epoch'] = 10000
params['RGB'] = 1

params['original_dim'] = 28
params['filter_size'] = 5
params['n_features'] = 16
params['s_dim'] = 10
params['l_size'] = 64
params['l_size2'] =64
params['latent_dim'] = 16
params['dp'] = 0.5
params['s_clss'] = 10
params['mseloss'] = True

#Need to calibrate based on AdvNet success:
params['diluteDist'] = 4 
params['dilutedDistEpoch'] = 4

curtime = strftime("%d%m%y_%H%M%S", localtime())
log_filename = params['res_path'] +'/'+ curtime + '_train_log.txt'
weihts_filename = params['res_path'] +'/'+ curtime + '_weights.h5'
outfile=open(log_filename,'a')

#Implement data load here
(x_train, y_train), (x_test, y_test) = load_data()#mnist.load_data()
print '\n Xtrain shape: ' ,x_train.shape
print 'Ytrain shape: ' ,y_train.shape
print 'Xtest shape: ' ,x_test.shape
print 'ytest shape: ' ,y_test.shape

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
ma = np.mean(x_train)
sa = np.std(x_train)
x_train = (x_train-ma)/sa
x_test = (x_test-ma)/sa
x_train = np.reshape(x_train, (x_train.shape[0], 1, params['original_dim'], params['original_dim']))
x_test = np.reshape(x_test, (x_test.shape[0], 1, params['original_dim'], params['original_dim']))

Y_train = np_utils.to_categorical(y_train, params['s_clss'])
Y_test = np_utils.to_categorical(y_test, params['s_clss'])

print 'Xtrain new shape: ' ,x_train.shape
print 'Ytrain new shape: ' ,Y_train.shape

permInds = np.random.permutation(x_train.shape[0])
x_valid = x_train[permInds[:2000],:,:,:]
Y_valid = Y_train[permInds[:2000],:]
y_valid = y_train[permInds[:2000]]

x_train = x_train[permInds[2000:],:,:,:]
Y_train = Y_train[permInds[2000:],:]
y_train = y_train[permInds[2000:]]

maxSamples = (np.floor(x_valid.shape[0]/params['batch_size'])*params['batch_size']).astype(np.int64)
x_valid = x_valid[:maxSamples,:,:,:]
Y_valid = Y_valid[:maxSamples,:]
y_valid = y_valid[:maxSamples]

print 'Xvalid new shape: ' ,x_valid.shape
print 'Yvalid new shape: ' ,Y_valid.shape

#loss weights - Need to calibrate so that both losses are effective:
params['recweight'] = 5
params['advweight'] = -1
net = DisAdvNet2.DisAdvNet2(params,outfile,weihts_filename,True)

#Stage 1 - S net training - make sure we acheive high accuracy (for mnist 99%)
hist = net.Snet.fit(x_train, Y_train,shuffle=True,epochs=40,batch_size=params['batch_size'],
                                        verbose=2,validation_data=(x_train, Y_train),callbacks=[net.checkpointer])


S_train  = net.EncS.predict(x_train,batch_size=100)
preds_s = net.Sclsfr.predict(S_train,batch_size=100)
accS = np.average(np.argmax(preds_s,1)==np.argmax(Y_train,1))
S_valid  = net.EncS.predict(x_valid,batch_size=100)
preds_s = net.Sclsfr.predict(S_valid,batch_size=100)
accS_val = np.average(np.argmax(preds_s,1)==np.argmax(Y_valid,1))
print accS,accS_val

loadS = ''
S_train = net.EncS.predict(x_train,batch_size=params['batch_size'])
S_valid = net.EncS.predict(x_valid,batch_size=params['batch_size'])
loss = net.train(params['nb_epoch'],x_train,Y_train,loadS=loadS,Strain=S_train,X_valid=x_valid,Y_valid=Y_valid)

outfile.close()  
