import os
import sys
import argparse
import glob
import shutil
import time
import json
import numpy as np
import torch
import torch.nn as nn
from yacs.config import CfgNode as CN
import _init_path
from dataset import EMGDataloader
# custom
from dataset import get_dataloader_class
from networks import get_network
from utils.trainer import Trainer
from utils.saver import save_checkpoint, save_result


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Jester Training using JPEG')
    parser.add_argument('--config', '-c', help='json config file path')
    parser.add_argument('--modelName', help='name of nn model')
    parser.add_argument('--modality', help='modality to train(EMG, US or USEMG)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.config) as cfg_file:
        cfg = CN.load_cfg(cfg_file)
        opts = [
            'ModelConfig.model_name', args.modelName, 
            'ModelConfig.modality', args.modality
            ]
        cfg.merge_from_list(opts)
        print(cfg)
        print('Successfully loading the config file....')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ModelConfig = cfg.ModelConfig
    paths_data = sorted(glob.glob('./data/'+ModelConfig.modality+'/*.txt'))
    class_dataloader = get_dataloader_class(ModelConfig.modality)
    results = np.zeros((len(paths_data), 4))
    for idx_subject in range(len(paths_data)):
        for cross_val in range(4): # 4-fold
            print(f"subject: {idx_subject}, cv: {cross_val}")
            # load dataloader
            print("Start loading the dataloader....")
            DataConfig = cfg.DatasetConfig
            train_loader, valid_loader = class_dataloader.get_dataloader(paths_data[idx_subject], cross_val, DataConfig)
            print('Finish loading the dataloader....')
            # load network
            with open(ModelConfig.model_arch[ModelConfig.model_name]) as data_file:
                cfg_model = CN.load_cfg(data_file)
                cfg_model = cfg_model[ModelConfig.modality]
            model = get_network(ModelConfig.model_name, cfg_model)
            model.to(device)
            # load checkpoint
            OutputConfig = cfg.OutputConfig
            checkpoint = torch.load(os.path.join(OutputConfig.dir_weights, ModelConfig.model_name, 
                    f'{ModelConfig.model_name}_{ModelConfig.modality}_s{idx_subject}_cv{cross_val}.pth.tar'))
            model.load_state_dict(checkpoint['state_dict'])
            # define criterion, optimizer, scheduler
            OptimizerConfig = cfg.OptimizerConfig
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=OptimizerConfig.lr, amsgrad =True)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150,180], gamma=0.5)
            # start training
            trainer = Trainer(train_loader, valid_loader, model, device, criterion, optimizer, print_freq=100)
            valid_loss, valid_acc = trainer.validate(eval_only=False)
            results[idx_subject, cross_val] = valid_acc
    save_result(results, os.path.join(OutputConfig.dir_results, ModelConfig.model_name), ModelConfig.model_name+ f'_{ModelConfig.modality}.txt')
    print('acc s/cv:\n', results)
    print('acc-avg-s:\n', np.mean(results, 1))
    print('acc-avg-total:\n', np.mean(np.mean(results, 1)))



if __name__ == "__main__":
    main()
    