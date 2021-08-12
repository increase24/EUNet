# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import math
import keras
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization,concatenate, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

#batch_size=32
path_allData = sorted(glob.glob('../../../../dataset/*.txt'))
path_processedUSData = sorted(glob.glob('../../../../processedUSData/*.txt'))

def LoadData(index_sub,group_test):
    #read raw EMG data
    df_emgData=pd.read_csv(path_allData[2*index_sub],sep=',',header = None)
    df_emgData=df_emgData.iloc[:,4:8]
    #df_emgData.columns = [0,1,2,3]
    df_emgData.rename(columns={4:0,5:1,6:2,7:3}, inplace=True)
    #normalization，各通道各自做标准化
    df_emgData = (df_emgData-df_emgData.mean())/(df_emgData.std())
    _max=max(abs(df_emgData.max(axis=0).max()),abs(df_emgData.min(axis=0).min()))
    df_emgData= df_emgData/_max
    df_emgData=df_emgData.append(pd.DataFrame(np.zeros((156,4))),ignore_index=True)
    data_org = np.zeros((10000,4,256,1))
    for i in range(data_org.shape[0]):
        data_org[i,:,:,:] = df_emgData.iloc[100*i:100*i+256,:].values.transpose().reshape([1,4,256,1])
    #exact 4800 valid samples from raw 10000 samples
    indexs_list=[]
    for i in range(8): # 8 trial
        for j in range(20): # 20 motions
            indexs = [i for i in range(i*1250+j*50+10,i*1250+j*50+40)]
            indexs_list=indexs_list+indexs
    data_org=data_org[indexs_list,:,:,:]

    #split train and test dataset
    group_train=[0,1,2,3]
    group_train.remove(group_test)
    data_train_emg = data_org[np.r_[group_train[0]*1200:(group_train[0]+1)*1200,
                                       group_train[1]*1200:(group_train[1]+1)*1200,
                                       group_train[2]*1200:(group_train[2]+1)*1200],:,:,:]
    data_test_emg = data_org[group_test*1200:(group_test+1)*1200,:,:,:]

    
    #read processed ultrasound data
    df_usData=pd.read_csv(path_processedUSData[index_sub],sep=',',header = None) #(,960*4)
    #normalization，各通道各自做标准化
    df_usData = (df_usData-df_usData.mean())/(df_usData.std())
    _max=max(abs(df_usData.max(axis=0).max()),abs(df_usData.min(axis=0).min()))
    df_usData= df_usData/_max
    data_org = df_usData.values.reshape((4800,4,960,1))
    labels = np.linspace(0,19,20, endpoint=True, dtype=int)
    labels = np.tile(labels, (30,1))
    labels= labels.reshape(-1,1,order = 'F')
    labels = np.tile(labels, (8,1))
    #split train and test dataset
    group_train=[0,1,2,3]
    group_train.remove(group_test)
    data_train_us = data_org[np.r_[group_train[0]*1200:(group_train[0]+1)*1200,
                                       group_train[1]*1200:(group_train[1]+1)*1200,
                                       group_train[2]*1200:(group_train[2]+1)*1200],:,:,:]
    data_test_us = data_org[group_test*1200:(group_test+1)*1200,:,:,:]
    
    
    #generate labels for train and test
    labels = np.linspace(0,19,20, endpoint=True, dtype=int)
    labels = np.tile(labels, (30,1))
    labels= labels.reshape(-1,1,order = 'F')
    labels = np.tile(labels, (8,1))
    label_train = labels[np.r_[group_train[0]*1200:(group_train[0]+1)*1200,
                                           group_train[1]*1200:(group_train[1]+1)*1200,
                                           group_train[2]*1200:(group_train[2]+1)*1200],:]
    label_test = labels[group_test*1200:(group_test+1)*1200,:]
    
    return data_train_emg,data_train_us,label_train,data_test_emg,data_test_us,label_test




# 对labels进行one-hot编码
def convert_to_one_hot(Y, C):
    Y_onehot = np.eye(C,dtype=np.int16)[Y.reshape(-1)]
    return Y_onehot

#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type,filename):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig(filename)
        plt.show()

def CNN(input_shape_emg=(4,256,1),input_shape_us=(4,960,1), classes=20):
    X_input_emg = Input(input_shape_emg)
    X1 = Conv2D(filters=32, kernel_size=(3,16), strides=(1,1), activation='relu', padding='same')(X_input_emg)
    X1 = MaxPooling2D((1,16))(X1)
    X1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(X1)
    X1 = MaxPooling2D((1,2))(X1)
    X1 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu',padding='valid')(X1)
    X1 = Flatten(name='flatten_emg')(X1)
    X1 = Dropout(0.5)(X1)
    X1 = Dense(128,activation='relu')(X1)
    
    X_input_us = Input(input_shape_us)
    X2 = Conv2D(filters=32, kernel_size=(3,16), strides=(1,1), activation='relu', padding='same')(X_input_us)
    X2 = MaxPooling2D((1,16))(X2)
    X2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(X2)
    X2 = MaxPooling2D((1,2))(X2)
    X2 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu',padding='valid')(X2)
    X2 = Flatten(name='flatte_us')(X2)
    X2 = Dropout(0.5)(X2)
    X2 = Dense(128,activation='relu')(X2)
    
    y = concatenate([X1,X2],axis=1)
#    y = Dropout(0.5)(y)
#    y = Dense(128,activation='relu')(y)
    y = Dropout(0.2)(y) # remove rate
    y = Dense(classes, activation='softmax')(y)
    model = Model(inputs=[X_input_emg,X_input_us], outputs=y)
    return model

#model = CNN()
#model.summary()

from keras.callbacks import ModelCheckpoint, EarlyStopping
result_cnn_us = np.zeros((8,4))
import time
for index_sub in range(8): # 8 subjects
    print("==========================")
    print("=======subject_{}=========".format(index_sub))
    for group_test in range(4):
        print("======trial_{}======".format(group_test))
        start = time.time()
        model = CNN()
        X_train_emg, X_train_us, y_train, X_test_emg, X_test_us, y_test= LoadData(index_sub, group_test)
        y_train = convert_to_one_hot(y_train,20)
        y_test = convert_to_one_hot(y_test,20)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = LossHistory() # 创建一个history实例
        
        best_weights_filepath = './best_weights.h5'
#        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                              patience=5, min_lr=0.00005)
        earlyStopping=EarlyStopping(monitor='val_acc', patience=50, verbose=1, mode='max')
        saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, \
                                        save_best_only=True, mode='max')

        model.fit([X_train_emg,X_train_us], y_train, epochs=200, validation_data=([X_test_emg,X_test_us], y_test),batch_size=32,\
                  callbacks=[history,saveBestModel,earlyStopping])
        #reload best weights
        model.load_weights(best_weights_filepath)
        
        preds_train = model.evaluate([X_train_emg,X_train_us], y_train)
        print("Train Loss = " + str(preds_train[0]))
        print("Train Accuracy = " + str(preds_train[1]))

        preds_test  = model.evaluate([X_test_emg,X_test_us], y_test)
        print("Test Loss = " + str(preds_test[0]))
        print("Test Accuracy = " + str(preds_test[1]))

        end = time.time()
        print("time:",end-start)
        result_cnn_us[index_sub, group_test] = preds_test[1]
        history.loss_plot('epoch',"./locc-acc/conv2d_S{}E{}.png".format(index_sub,group_test))
np.savetxt("./results/result_conv2d_usemg.txt",result_cnn_us)


