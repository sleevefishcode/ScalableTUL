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
from utils import  accuracy_1, accuracy_5
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
from model import calculate_similarity
import warnings

from predictor import PredictorLayer

warnings.filterwarnings("ignore")
def cos_loss(output,target):
    device=output.device
    cosine_loss = nn.CosineEmbeddingLoss(margin=0)
    y=torch.ones(output.size(0)).to(device)
    loss=cosine_loss(output,target,y)
    return loss
def compute_loss(output,target):
  
    return cos_loss(output,target)

def test_model(dataset,test_dataset,user_embedding, model,predictor_layer, devices, args, logger,wandb_log):
    checkpoint=torch.load('temp/'+ dataset +str(args.seed)+'final_best_checkpoint.pt')
    predictor_layer_checkpoint=torch.load('temp/'+ dataset +str(args.seed)+'final_predictor_best_checkpoint.pt')
    model.load_state_dict(checkpoint['net'])
    predictor_layer.load_state_dict(predictor_layer_checkpoint['predictor_layer'])
    print("Final alph value:",  model.module.alph.item())
    model.eval()
    predictor_layer.eval()
    test_dataloader = get_dataloader(traj_dataset = test_dataset, load_datatype='test', batch_size=args.batch_size)
    loss_test_list_1, y_predict_list_1, y_true_list_1, acc1_list, acc5_list_1 = [], [], [], [], []
    
    with torch.no_grad():
        for poi_seq_1, category_seq_1, hour_seq_1, time_seq_1, current_len_1,one_batch_label in test_dataloader:
            
            poi_seq_1, category_seq_1, hour_seq_1, time_seq_1 = poi_seq_1.to(devices[0]), category_seq_1.to(devices[0]), hour_seq_1.to(devices[0]), time_seq_1.to(devices[0])
            current_len_1, one_batch_label = current_len_1.to(devices[0]) ,one_batch_label.to(devices[0])          
            
  
            user_embedding_on_device = {}

            for user_id, embedding_vector in user_embedding.items():
                tensor_embedding = torch.tensor(embedding_vector).to(devices[0])
                user_embedding_on_device[user_id] = tensor_embedding
      
            one_batch_label_embedding_on_device = torch.stack([user_embedding_on_device[str(label.item())] for label in one_batch_label])
            output = model(poi_seq_1, category_seq_1, hour_seq_1, current_len_1, time_seq_1)
     
            
            test_prediction_output=predictor_layer(output)
       
            id_output, id5_output=calculate_similarity(user_embedding_on_device,test_prediction_output)            
      
            y_predict_list_1.extend(id_output)
            id_output=torch.tensor(id_output, dtype=torch.long).to(devices[0])
            id5_output=torch.tensor(id5_output, dtype=torch.long).to(devices[0])           
            
      
            y_true_list_1.extend(one_batch_label.cpu().numpy().tolist())
            acc1=accuracy_1(id_output, one_batch_label).cpu().numpy()
            acc1_list.extend(acc1)
            acc5=accuracy_5(id5_output, one_batch_label).cpu().numpy()
          
            acc5_list_1.extend(acc5)
      
            loss_1 = compute_loss(test_prediction_output, one_batch_label_embedding_on_device)
            loss_test_list_1.append(loss_1.item())
      
        macro_p = precision_score(y_true_list_1, y_predict_list_1, average='macro')
        macro_r = recall_score(y_true_list_1, y_predict_list_1, average='macro')
        macro_f1 = f1_score(y_true_list_1, y_predict_list_1, average='macro')
        micro_p = precision_score(y_true_list_1, y_predict_list_1, average='micro')
        micro_r = recall_score(y_true_list_1, y_predict_list_1, average='micro')
        micro_f1 = f1_score( y_true_list_1, y_predict_list_1, average='micro')
        wandb_log.log(
                {
                    "test loss": np.mean(loss_test_list_1),
                    "test acc@1":np.mean(acc1_list),
                    "test acc@5":np.mean(acc5_list_1),
                    "test macro_p":macro_p,
                    "test macro_r":macro_r,
                    "test macro_f1":macro_f1,
                     "test micro_p":micro_p,
                    "test micro_r":micro_r,
                    "test micro_f1":micro_f1,
                }
            )         
        output_content = "Test \t loss:{:.6f} acc@1:{:.6f} acc@5:{:.6f} macro_p:{:.6f} macro_r:{:.6f} macro_f1:{:.6f} micro_p:{:.6f} micro_r:{:.6f} micro_f1:{:.6f}"
        logger.info(output_content.format(np.mean(loss_test_list_1), np.mean(acc1_list),np.mean(acc5_list_1), macro_p, macro_r, macro_f1,micro_p, micro_r, micro_f1))

       
