import pandas as pd
import numpy as np
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .USDataset import USDataset

def get_dataloader(data_path, group_test, DataConfig):
    df_usData=pd.read_csv(data_path,sep=',',header = None) #(n_sample,960*4)
    # normalization
    df_usData = (df_usData-df_usData.mean())/(df_usData.std())
    _max=max(abs(df_usData.max(axis=0).max()),abs(df_usData.min(axis=0).min()))
    df_usData= df_usData/_max
    data_org = df_usData.values.reshape((4800,1,4,960))
    data_org = np.concatenate([data_org, np.zeros((4800,1,4,64))], axis = -1)
    data_org = data_org.astype('float32')
    labels = np.linspace(0,19,20, endpoint=True, dtype=int)
    labels = np.tile(labels, (30,1))
    labels= labels.reshape(-1,1,order = 'F')
    labels = np.tile(labels, (8,1))
    #split train and test dataset (4-fold)
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
    train_data = USDataset(data_train, label_train)
    valid_data = USDataset(data_test, label_test)
    train_loader = DataLoader(train_data, batch_size = DataConfig.batch_size, shuffle = True, num_workers = DataConfig.num_workers, 
        pin_memory = True, drop_last = True)
    valid_loader = DataLoader(valid_data, batch_size = DataConfig.batch_size, shuffle = False, num_workers = DataConfig.num_workers,
        pin_memory = True, drop_last = False)
    return train_loader, valid_loader

if __name__ == "__main__":
    paths_USData = sorted(glob.glob('./data/us/*.txt'))
    idx_subject = 0
    train_loader, valid_loader = get_dataloader(paths_USData[idx_subject], 3)
    samples, labels = iter(valid_loader).next()
    print(samples.shape, labels.shape)
    fig = plt.figure()
    ax1 = fig.add_subplot(4,1,1)
    ax1.plot(samples[0][0][0].numpy())
    ax2 = fig.add_subplot(4,1,2)
    ax2.plot(samples[0][0][1].numpy())
    ax3 = fig.add_subplot(4,1,3)
    ax3.plot(samples[0][0][2].numpy())
    ax4 = fig.add_subplot(4,1,4)
    ax4.plot(samples[0][0][3].numpy())
    plt.show()

