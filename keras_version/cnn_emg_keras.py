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
path_allData = sorted(glob.glob('./data/emg/*.txt'))
    
def LoadData(index_sub,group_test):
    df_emgData=pd.read_csv(path_allData[index_sub],sep=',',header = None)
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
    labels = np.linspace(0,19,20, endpoint=True, dtype=int)
    labels = np.tile(labels, (30,1))
    labels= labels.reshape(-1,1,order = 'F')
    labels = np.tile(labels, (8,1))
    #split train and test dataset
    group_train=[0,1,2,3]
    group_train.remove(group_test)
    data_train = data_org[np.r_[group_train[0]*1200:(group_train[0]+1)*1200,
                                       group_train[1]*1200:(group_train[1]+1)*1200,
                                       group_train[2]*1200:(group_train[2]+1)*1200],:,:,:]
    data_test = data_org[group_test*1200:(group_test+1)*1200,:,:,:]
    label_train = labels[np.r_[group_train[0]*1200:(group_train[0]+1)*1200,
                                           group_train[1]*1200:(group_train[1]+1)*1200,
                                           group_train[2]*1200:(group_train[2]+1)*1200],:]
    label_test = labels[group_test*1200:(group_test+1)*1200,:]
    
    return data_train,label_train,data_test,label_test

# 提取随机的batch进行训练
def random_mini_batches(X, Y, mini_batch_size=10):

    m = X.shape[0]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m / mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


#def weight_variable(shape):
#    # 用正态分布来初始化权值
#    initial = tf.truncated_normal(shape, stddev=0.1)
#    return tf.Variable(initial)

#def bias_variable(shape):
#    # 本例中用relu激活函数，所以用一个很小的正偏置较好
#    initial = tf.constant(0.1, shape=shape)
#    return tf.Variable(initial)

#def batch_normalization(bnin):
#    batch_mean,batch_var=tf.nn.moments(bnin,[0,1,2],keep_dims=True)
#    shift=tf.Variable(tf.zeros([batch_mean]))
#    scale=tf.Variable(tf.ones([batch_mean]))
#    BN_out=tf.nn.batch_normalization(bnin,batch_mean,batch_var,shift,scale,epsilon)
#    return BN_out


def create_placeholders(n_H0,n_W0,n_C0,n_y):
    X=tf.placeholder('float',[None,n_H0,n_W0,n_C0])
    Y=tf.placeholder('float',[None,n_y])
    return X,Y

# 对数据做一个归一化处理，可以提速
def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2]) #对每一个kernel对应的batch_size个图，求它们所有像素的mean和varience
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

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

def CNN(input_shape=(4,256,1), classes=20):
    X_input = Input(input_shape)

    X = Conv2D(filters=32, kernel_size=(3,20), strides=(1,1), activation='relu', padding='same')(X_input)
    X = MaxPooling2D((1,20))(X)

    X = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(X)
    X = MaxPooling2D((1,2))(X)

    X = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu',padding='valid')(X)
#    X = MaxPooling2D((2,2))(X)
#    
#    X = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu',padding='valid')(X)

    X = Flatten(name='flatten')(X)
    X = Dropout(0.5)(X)
    X = Dense(128,activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax')(X)
    model = Model(inputs=X_input, outputs=X)
    return model

#model = CNN()
#model.summary()

from keras.callbacks import ModelCheckpoint, EarlyStopping
result_cnn_emg = np.zeros((8,4))
import time
for index_sub in range(8): # 8 subjects
    print("==========================")
    print("=======subject_{}=========".format(index_sub))
    for group_test in range(4):
        print("======trial_{}======".format(group_test))
        start = time.time()
        model = CNN()
        X_train, y_train, X_test, y_test= LoadData(index_sub, group_test)
        y_train = convert_to_one_hot(y_train,20)
        y_test = convert_to_one_hot(y_test,20)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = LossHistory() # 创建一个history实例
        
        best_weights_filepath = './best_weights.h5'
#        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                              patience=5, min_lr=0.00005)
        earlyStopping=EarlyStopping(monitor='val_accuracy', patience=50, verbose=1, mode='max')
        saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_accuracy', verbose=1, \
                                        save_best_only=True, mode='max')

        model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test),batch_size=32,\
                  callbacks=[history,saveBestModel,earlyStopping])
        #reload best weights
        model.load_weights(best_weights_filepath)
        
        preds_train = model.evaluate(X_train, y_train)
        print("Train Loss = " + str(preds_train[0]))
        print("Train Accuracy = " + str(preds_train[1]))

        preds_test  = model.evaluate(X_test, y_test)
        print("Test Loss = " + str(preds_test[0]))
        print("Test Accuracy = " + str(preds_test[1]))

        end = time.time()
        print("time:",end-start)
        result_cnn_emg[index_sub, group_test] = preds_test[1]
        #history.loss_plot('epoch',"./outputs/loss-acc/emg_conv2d_S{}E{}.png".format(index_sub,group_test))
np.savetxt("./outputs/results/result_conv2d_emg.txt",result_cnn_emg)

