from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.layers import concatenate
from keras.optimizers import SGD, Adam, Adadelta,RMSprop
from keras.layers.core import Activation,Dense,Dropout,Flatten,Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D,UpSampling2D,AveragePooling2D
from keras.layers import Input, Lambda
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Cropping2D

import numpy as np

class DisAdvNet2(object):
    
    def log_results(self,log_file,res_dict,debug=False,display_on=False):
        for key,val in res_dict.iteritems():
            #print '%s: %.4f' % (key,val)
            if self.save_files: log_file.write(key + ' ' + str(val) + ' ')
	    if display_on:
                print key + ' ' + str(val) + '\n'
        #print ''
        if self.save_files: log_file.write('\n')
        if self.save_files: log_file.flush()
    def log_weights(self,cur_loss):
        if self.save_files:
            self.min_loss = cur_loss
            self.DistNet.save(self.weights_file + '_full_weights.h5')
            self.EncZ.save(self.weights_file + '_encZ_weights.h5')
            self.Dec.save(self.weights_file + '_decS_weights.h5')
            self.Snet.save(self.weights_file + '_Sfreeze_weights.h5')
            self.Adv.save(self.weights_file + '_adv_weights.h5')
            self.AdvNet.save(self.weights_file + '_advNet_weights.h5')
            print 'saving files'
        
    def __init__(self, params,outfile,weights_file,save_files):
        batch_size=params['batch_size']
        original_dim=params['original_dim']
        latent_dim=params['latent_dim']
        s_dim=params['s_dim']
        l_size=params['l_size']
        filter_size = params['filter_size']
        n_features = params['n_features']
        s_clss = params['s_clss']
        init = params['Init']#eval(params['Init']+'()')
        if 'LeakyReLU' in params['Activation']:
            act = eval(params['Activation']+'(0.1)')
        else:
            act = Activation(params['Activation'])
        actstr = params['Activation']
        use_bn = params['BN']
        dp = params['dp']   
        if 'RGB' in params:
            RGB=params['RGB']
        else:
            RGB=1

        if 'mseloss' in params and params['mseloss']==True:
            mseloss = 'mse'
        else:
            mseloss = 'binary_crossentropy'
        encS_bn = params['EncBN']
        encZ_bn = params['EncBN']
        dec_bn = params['DecBN']
        
        def add_dense_layer(inp,dim,out_dp=0):
            h = Dense(dim,init=init)(inp)
            if use_bn:
                h = BatchNormalization(mode=2)(h)#mode=2
            h = act(h)
            if out_dp==1 and dp>0:
                h = Dropout(dp)(h)
            return h
        
        def clipping(args):
            vals = args
            return K.clip(vals,-30,30)
            
        def add_conv_layer(inp,n_features,filter_size,bn=False,actconv='relu',stride=True,dilation=False,maxpool=False,upsamp = False,avgpool=False):
            if stride:
                h = Conv2D(n_features, kernel_size=(filter_size, filter_size),strides=(2,2), padding='same')(inp)
            elif dilation:
                h = Conv2D(n_features, kernel_size=(filter_size, filter_size),dilation_rate=(2,2), padding='same')(inp)
            else:
                h = Conv2D(n_features, kernel_size=(filter_size, filter_size), padding='same')(inp)
            if upsamp:
                h = UpSampling2D((2, 2))(h)
            if bn:
                h = BatchNormalization(axis = 1)(h)#mode=2
            h = actconv(h)
            if maxpool:
                h = MaxPooling2D((2, 2), padding='same')(h)
            if avgpool:
                h = AveragePooling2D((2, 2), padding='same')(h)
            return h
        
        #shp = (1,original_dim,original_dim)
        input_img = Input(batch_shape=(batch_size, RGB, original_dim, original_dim))
        #input_img = Input(shape=shp)
        
        #Each part of the network has to be adjusted to the complexity of the data
        #here I use simple architecture with 3 convolutional layers
        #Enc Z ########################################
        x = add_conv_layer(input_img,n_features,filter_size,bn=encZ_bn,actconv=act,stride=True,dilation=False,maxpool=False,upsamp = False,avgpool=False)
        x = add_conv_layer(x,n_features/2,filter_size,bn=encZ_bn,actconv=act,stride=True,dilation=False,maxpool=False,upsamp = False,avgpool=False)
        x = add_conv_layer(x,n_features/2,filter_size,bn=encZ_bn,actconv=act,stride=False,dilation=False,maxpool=False,upsamp = False,avgpool=False)
        
        h = Flatten()(x)
        h = add_dense_layer(h,256,out_dp=0)
        z_mean = Dense(latent_dim,kernel_initializer=init, activation=actstr)(h)
        #z_log_var = Dense(latent_dim,kernel_initializer=init, activation=actstr)(h)
        self.EncZ = Model(inputs = input_img, output=[z_mean])#,z_log_var])
            
        
        #Enc S ########################################
        x = add_conv_layer(input_img,n_features,filter_size,bn=encS_bn,actconv=act,stride=True,dilation=False,maxpool=False,upsamp = False,avgpool=False)
        x = add_conv_layer(x,n_features/2,filter_size,bn=encS_bn,actconv=act,stride=True,dilation=False,maxpool=False,upsamp = False,avgpool=False)
        x = add_conv_layer(x,n_features/2,filter_size,bn=encS_bn,actconv=act,stride=False,dilation=False,maxpool=False,upsamp = False,avgpool=False)
        divShape = 4
        h = Flatten()(x)
        h = add_dense_layer(h,256,out_dp=0)
        s = Dense(s_dim,kernel_initializer=init, activation=actstr)(h)  
        self.EncS = Model(inputs = input_img, outputs=s)
        

        #Dec ########################################
        in_z = Input(batch_shape=(batch_size, latent_dim))
        in_s = Input(batch_shape=(batch_size, s_dim))
        
        inz_s = concatenate(inputs = [in_z, in_s],axis=1)#([in_z, in_s])

        x = Dense(n_features*original_dim*original_dim/(2*divShape*divShape),kernel_initializer=init, activation=actstr)(inz_s)
        x = Reshape((n_features/2,original_dim/divShape,original_dim/divShape))(x)
        x = add_conv_layer(x,n_features/2,filter_size,bn=dec_bn,actconv=act,stride=False,upsamp = True)
        x = add_conv_layer(x,n_features/2,filter_size,bn=dec_bn,actconv=act,stride=False,upsamp = True)
        x = add_conv_layer(x,n_features,filter_size,bn=dec_bn,actconv=act,stride=False,upsamp = False)
        
        if mseloss == 'binary_crossentropy':
            decoder_h = Conv2D(RGB, kernel_size=(filter_size, filter_size), activation='sigmoid', padding='same')(x)
        else:
            decoder_h = Conv2D(RGB, kernel_size=(filter_size, filter_size), padding='same')(x)
 
        self.Dec = Model([in_z,in_s],[decoder_h])#,x_decoded_log_std])#logpxz

        #Adv ########################################
        adv_h = add_dense_layer(in_z,l_size,out_dp=0)
        adv_h = add_dense_layer(adv_h,l_size,out_dp=0)
        adv_h = add_dense_layer(adv_h,l_size,out_dp=0)
        adv_h = Dense(s_clss,kernel_initializer=init)(adv_h)
        out = Activation('softmax')(adv_h)
        
        self.Adv = Model(in_z,out)
        ########################################
        
        #Sclsfr ########################################
        hclsfr = add_dense_layer(in_s,l_size,out_dp=0)
        hclsfr = add_dense_layer(hclsfr,l_size,out_dp=0)
        hclsfr = Dense(s_clss,kernel_initializer=init)(hclsfr)
        outhclsfr= Activation('softmax')(hclsfr)
        
        self.Sclsfr = Model(in_s,outhclsfr)
        ########################################
        
        print 'building enc...'
        x1 = Input(batch_shape=(batch_size,RGB, original_dim, original_dim))
        Z1in = Input(batch_shape=(batch_size, latent_dim))
        
        s1 = self.EncS(x1)
        z1_mean = self.EncZ(x1)#,z1_log_var
            
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
            return z_mean + K.exp(z_log_var / 2) * epsilon
        
        #z1 = Lambda(sampling, output_shape=(latent_dim,))([z1_mean, z1_log_var]) #VAE encoder
        z1 = z1_mean #Normal encoder 
                   
        print 'building dec...'
        x11 = self.Dec([z1,in_s])#,x11_log_std
        
        print 'building Sclsifier...'
        Sclsfr = self.Sclsfr(s1)
        
        print 'building Adv...'
        Adv1 = self.Adv(z1)
        Adv1_netAdv = self.Adv(Z1in)
        
        recweight = params['recweight'] if 'recweight' in params else 0.5
        advweight = params['advweight'] if 'advweight' in params else -0.1
            
        print 'compile...'
        self.DistNet = Model([x1,in_s], [x11,Adv1])#[x11,z1,Adv1])
        self.freeze_unfreeze_Adv(False)
        self.freeze_unfreeze_Enc(True)
        self.freeze_unfreeze_Dec(True)
        self.freeze_unfreeze_Spart(False)
        opt = Adam(lr=0.0001, beta_1=0.5)
        self.DistNet.compile(optimizer=opt, loss=[mseloss,'categorical_crossentropy'],loss_weights=[recweight,advweight])
        
        self.Snet = Model(x1, Sclsfr)
        self.freeze_unfreeze_Adv(False)
        self.freeze_unfreeze_Enc(False)
        self.freeze_unfreeze_Dec(False)
        self.freeze_unfreeze_Spart(True)
        opt = Adam(lr=0.0001, beta_1=0.9)
        self.Snet.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
    
        self.AdvNet = Model(Z1in,Adv1_netAdv)
        self.freeze_unfreeze_Enc(False)
        self.freeze_unfreeze_Dec(False)
        self.freeze_unfreeze_Adv(True)
        self.freeze_unfreeze_Spart(False)
        opt = Adam(lr=0.00002, beta_1=0.9)
        self.AdvNet.compile(optimizer='sgd', loss='categorical_crossentropy')#loss#adv_loss)  
        
        self.params = params
        self.outfile = outfile
        self.save_files = save_files
        if self.save_files:
            self.log_results(self.outfile,params,debug=False)
            self.weights_file = weights_file
            self.checkpointer = ModelCheckpoint(filepath=(self.weights_file + '_Snet_weights.h5'), monitor='val_acc', verbose=1, save_best_only=True)

    def train(self,nb_epoch,X_train,Y_train,loadS,Strain,X_valid,Y_valid):
        
        train_params={}
        train_params['X_train_len'] = X_train.shape[0]
        train_params['data_cols'] = X_train.shape[1]
        
        self.log_results(self.outfile,train_params,debug=False,display_on=True)
        if len(loadS)>0:
            self.Snet.load_weights(loadS + '_Snet_weights.h5')
        else:
            self.Snet.load_weights(self.weights_file + '_Snet_weights.h5')
      
        self.min_loss = 100000000
        self.adv_loss=100000000
        batch_size = self.params['batch_size']
        stopat=100000
        diluteDist = self.params['diluteDist']
        dilutedDistEpoch = self.params['dilutedDistEpoch']
        trainDistEpoch = dilutedDistEpoch-1
        for e in range(nb_epoch):
            trainDistEpoch = trainDistEpoch+1

            trainDist=diluteDist

            perm_inds = np.random.permutation(range(X_train.shape[0]))
            X_trainperm = X_train[perm_inds,:,:,:]
            Y_trainperm = Y_train[perm_inds,:]
            S_trainperm = Strain[perm_inds,:]
            for (first, last) in zip(range(0, min(X_trainperm.shape[0]-batch_size,stopat-batch_size), batch_size),
                                     range(batch_size, min(X_trainperm.shape[0],stopat), batch_size)):
                X1 = X_trainperm[first: last, :,:,:]
                Slbls = Y_trainperm[first: last,:]
                Svec = S_trainperm[first: last,:]
                if trainDistEpoch == dilutedDistEpoch:
                    if trainDist==diluteDist:
                        loss0batch= self.DistNet.train_on_batch([X1,Svec],[X1,Slbls])
                        trainDist = 0
                    trainDist =  trainDist+1
                    
                Z1 = self.EncZ.predict(X1,batch_size=self.params['batch_size'])
                loss1batch= self.AdvNet.train_on_batch(Z1,Slbls)
            
            maxSamples = (np.floor(X_train.shape[0]/self.params['batch_size'])*self.params['batch_size']).astype(np.int64)
            lossall = self.DistNet.evaluate([X_train[:maxSamples,:,:,:],Strain[:maxSamples,:]],[X_train[:maxSamples,:,:,:],Y_train[:maxSamples,:]], batch_size=batch_size, verbose=0)
            
            if trainDistEpoch == dilutedDistEpoch:
                trainDistEpoch = 0
                
            Z_val = self.EncZ.predict(X_valid,batch_size=self.params['batch_size'])
            predsZ_val = self.AdvNet.predict(Z_val,batch_size=self.params['batch_size'])
            accZ = np.average(np.argmax(predsZ_val,1)==np.argmax(Y_valid,1))
            print 'epoch: ' +str(e)+' accuracyZ '+str(round(accZ,3))+', total loss: ' + str(round(lossall[0],3))+', reconstruction loss: ' + str(round(lossall[1],3)) +', adv loss: ' + str(round(lossall[2],3)) 
                                    
        return lossall

    def predictEnc(self,X1):
        
        batch_size = self.params['batch_size']
       
        s1 = self.EncS.predict(X1,batch_size=batch_size)
        z1_m= self.EncZ.predict(X1,batch_size=batch_size)#z1_std
        z1_std=np.zeros(z1_m.shape)
            
        return z1_m,z1_std,s1#, z1_std, s1#z1_m_full, z1_std_full, s1_full
        
    def freeze_unfreeze_Adv(self,trainable = False):
        self.Adv.trainable = trainable
        for l in self.Adv.layers:
            l.trainable = trainable
    def freeze_unfreeze_Spart(self,trainable = False):
        self.EncS.trainable = trainable
        for l in self.EncS.layers:
            l.trainable = trainable 
        self.Sclsfr.trainable = trainable
        for l in self.Sclsfr.layers:
            l.trainable = trainable
    def freeze_unfreeze_Enc(self,trainable = False):
        self.EncZ.trainable = trainable        
        for l in self.EncZ.layers:
            l.trainable = trainable
    def freeze_unfreeze_Dec(self,trainable = False):
        self.Dec.trainable = trainable
        
        for l in self.Dec.layers:
            l.trainable = trainable