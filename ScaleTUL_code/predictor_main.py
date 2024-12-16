import time
import random
import logging
import argparse
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
from loaddataset import get_dataset, get_dataloader
from preprocess import split_dataset
from utils import EarlyStopping_acc, accuracy_1, accuracy_5, loss_with_plot
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
from model import calculate_similarity
from test import test_model
from loss import SupConLoss
import warnings
from projection import ProjectionLayer
from predictor import PredictorLayer



def cos_loss(output,target):
    device=output.device
    cosine_loss = nn.CosineEmbeddingLoss(margin=0)
    y=torch.ones(output.size(0)).to(device)
    loss=cosine_loss(output,target,y)
    return loss
    

def predictor(train_dataset, train_sampler,user_embedding ,valid_sampler, model,predictor_layer, optimizer, user_traj_train, devices, args, logger,wandb_log):
    avg_train_losses = []
    avg_valid_losses = []
    avg_valid_acc=[]

    early_stopping = EarlyStopping_acc(logger=logger, dataset_name=args.dataset, seed=args.seed,patience=5, verbose=True,delta=0)
    train_dataloader = get_dataloader(traj_dataset = train_dataset, load_datatype='train', batch_size=args.batch_size, sampler=train_sampler, user_traj_train=user_traj_train)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
    checkpoint=torch.load('temp/'+ args.dataset+str(args.seed) +'final_best_checkpoint.pt')
    model.load_state_dict(checkpoint['net'])
    user_embedding_on_device = {}
    

    for user_id, embedding_vector in user_embedding.items():
        tensor_embedding = torch.tensor(embedding_vector).to(devices[0])
        user_embedding_on_device[user_id] = tensor_embedding
    for epoch_idx in range(args.epochs+start_epoch):
        model.eval()
        predictor_layer.train()
        loss_train_list = []
        loss_train_MAE_list = []
        for batch_idx, (poi_seq, category_seq, hour_seq, time_seq, current_len, positive_poi, positive_category, positive_hour, positive_time, positive_current_len, one_batch_label) in enumerate(tqdm(train_dataloader)):
            poi_seq, category_seq, hour_seq, time_seq= poi_seq.to(devices[0]), category_seq.to(devices[0]), hour_seq.to(devices[0]), time_seq.to(devices[0])
            current_len, one_batch_label = current_len.to(devices[0]), one_batch_label.to(devices[0])
            one_batch_label_embedding_on_device = torch.stack([user_embedding_on_device[str(label.item())] for label in one_batch_label])
            with torch.no_grad():
                train_output = model(poi_seq, category_seq, hour_seq, current_len, time_seq)
            train_prediction_output=predictor_layer(train_output.detach())
        
            loss_train_MAE = cos_loss(train_prediction_output, one_batch_label_embedding_on_device)
            loss_train=loss_train_MAE
     

            optimizer.zero_grad()
            loss_train.backward()
     
            optimizer.step()
            loss_train_list.append(loss_train.item())
        wandb_log.log(
                        {                             
                            "predictor-train loss": np.mean(loss_train_list),
                        }
                    )
        output_content = "Train epoch:{} batch:{}     Train_loss:{:.6f}"               
        logger.info(output_content.format(epoch_idx, batch_idx ,np.mean(loss_train_list)))    
        model.eval()
        predictor_layer.eval()    
        gc.collect()
        torch.cuda.empty_cache()
        valid_dataloader = get_dataloader(traj_dataset = train_dataset, load_datatype='valid', batch_size=args.batch_size, sampler=valid_sampler,  user_traj_train=user_traj_train)
        loss_valid_list, y_predict_list, y_true_list, acc1_list, acc5_list = [], [], [], [], []
        loss_valid_MAE_list , loss_valid_con_list = [], []   
        with torch.no_grad():

            for batch_idx, (poi_seq, category_seq, hour_seq, time_seq, current_len,positive_poi, positive_category, positive_hour, positive_time, positive_current_len,one_batch_label) in enumerate(tqdm(valid_dataloader)):

                poi_seq, category_seq, hour_seq, time_seq= poi_seq.to(devices[0]), category_seq.to(devices[0]), hour_seq.to(devices[0]), time_seq.to(devices[0])
                current_len, one_batch_label = current_len.to(devices[0]), one_batch_label.to(devices[0])
                
                one_batch_label_embedding_on_device = torch.stack([user_embedding_on_device[str(label.item())] for label in one_batch_label])

                valid_output  = model(poi_seq, category_seq, hour_seq, current_len, time_seq)  
                val_prediction_output=predictor_layer(valid_output)
                id_output, id5_output=calculate_similarity(user_embedding_on_device,val_prediction_output)
                y_predict_list.extend(id_output)
                id_output=torch.tensor(id_output, dtype=torch.long).to(devices[0])
                id5_output=torch.tensor(id5_output, dtype=torch.long).to(devices[0])
             

                y_true_list.extend(one_batch_label.cpu().numpy().tolist())
                acc1=accuracy_1(id_output, one_batch_label).cpu().numpy()
                
                acc1_list.extend(acc1)

                acc5=accuracy_5(id5_output, one_batch_label).cpu().numpy()

                acc5_list.extend(acc5)

                loss_val_MAE = cos_loss(val_prediction_output, one_batch_label_embedding_on_device)
                loss_valid=loss_val_MAE
          

                
                loss_valid_list.append(loss_valid.item())
                
                
            macro_p = precision_score(y_true_list, y_predict_list, average='macro')
            macro_r = recall_score(y_true_list, y_predict_list, average='macro')
            macro_f1 = f1_score( y_true_list, y_predict_list, average='macro')
            micro_p = precision_score(y_true_list, y_predict_list, average='micro')
            micro_r = recall_score(y_true_list, y_predict_list, average='micro')
            micro_f1 = f1_score( y_true_list, y_predict_list, average='micro')
            wandb_log.log(
                    {   "predictor-epoch":epoch_idx,
                        "predictor-learnig_rate":optimizer.param_groups[0]['lr'],
                        "predictor-valid loss": np.mean(loss_valid_list),
                        "predictor-valid acc@1":np.mean(acc1_list),
                        "predictor-valid acc@5":np.mean(acc5_list),
                        "predictor-valid macro_p":macro_p,
                        "predictor-valid macro_r":macro_r,
                        "predictor-valid macro_f1":macro_f1,
                        "predictor-valid micro_p":micro_p,
                        "predictor-valid micro_r":micro_r,
                        "predictor-valid micro_f1":micro_f1,
                        
                    }
                )            
            output_content = "Valid epoch:{}   valid_loss:{:.6f} acc@1:{:.6f} acc@5:{:.6f} macro_p:{:.6f} macro_r:{:.6f} macro_f1:{:.6f} micro_p:{:.6f} micro_r:{:.6f} micro_f1:{:.6f}"
            logger.info(output_content.format(epoch_idx,np.mean(loss_valid_list), np.mean(acc1_list), np.mean(acc5_list),macro_p, macro_r, macro_f1,micro_p, micro_r, micro_f1))
        
        avg_train_losses.append(np.mean(loss_train_list))
        avg_valid_losses.append(np.mean(loss_valid_list))
        avg_valid_acc.append((np.mean(acc1_list)))
        checkpoint = {
            "predictor_layer": predictor_layer.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch_idx,
            'lr_scheduler': scheduler.state_dict()
        }
 
        early_stopping(avg_valid_acc[-1], model,checkpoint)
        if early_stopping.early_stop:
            logger.info('Early Stop!')
            break
        else:
            
            scheduler.step()
