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
from utils import EarlyStopping_loss, accuracy_1, accuracy_5, loss_with_plot
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
from model import LSTMEncoder, TulNet, single_Mamba,TemporalEncoding, TransformerTimeAwareEmbedding, LstmTimeAwareEmbedding
from test import test_model
from loss import SupConLoss
import warnings
from projection import ProjectionLayer
from predictor import PredictorLayer
from datetime import datetime
import pickle


from predictor_main import predictor
warnings.filterwarnings("ignore")
import os 
os.environ['CUDA_VISIBLE_DEVICES'] ='3'  

def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='ScaleTUL')
    parse.add_argument('--times', type=int, default=1, help='times of repeat experiment')
    parse.add_argument('--dataset', type=str, default="foursquare_mini", help='dataset for experiment')
    parse.add_argument('--epochs', type=int, default=100, help='Number of total epochs')
    parse.add_argument('--batch_size', type=int, default=512, help='Size of one batch')
    parse.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parse.add_argument('--temperature', type=float, default=0.1, help='contastive Temperature hyperparameter')

    parse.add_argument('--embed_size', type=int, default=512, help='Number of embeding dim')
    parse.add_argument('--num_heads', type=int, default=8, help='Number of heads')
    parse.add_argument('--num_layers', type=int, default=3, help='Number of EncoderLayer')
    parse.add_argument('--mask', type=float, default=0.9, help='Ratio of mask POI')
    parse.add_argument('--seed', type=int, default=2024, help='set of seed')
    args = parse.parse_args()
    return args


def getLogger(dataset):
    """[Define logging functions]

    Args:
        dataset ([string]): [dataset name]
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(filename='./log/'+dataset+'.log', mode='w')
    consoleHandler.setLevel(logging.INFO)

    consoleformatter = logging.Formatter("%(message)s")
    fileformatter = logging.Formatter("%(message)s")

    consoleHandler.setFormatter(consoleformatter)
    fileHandler.setFormatter(fileformatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    return logger

def try_all_gpus():
 
    devices = [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]
    return devices if devices else [torch.device('cpu')]


def cos_loss(output,target):
    device=output.device
    cosine_loss = nn.CosineEmbeddingLoss(margin=0)
    y=torch.ones(output.size(0)).to(device)
    loss=cosine_loss(output,target,y)
    return loss
def train_model(train_dataset, train_sampler,user_embedding ,valid_sampler, model,projection_layer,predictor_layer, optimizer, user_traj_train, devices, args, logger,wandb_log):
    avg_train_losses = []
    avg_valid_losses = []
    early_stopping = EarlyStopping_loss(logger=logger, dataset_name=args.dataset,seed= args.seed,patience=6, verbose=True,delta=0)
    train_dataloader = get_dataloader(traj_dataset = train_dataset, load_datatype='train', batch_size=args.batch_size, sampler=train_sampler, user_traj_train=user_traj_train,mask_ratio=args.mask)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.8)
    start_epoch=0

    criterion = SupConLoss(temperature=args.temperature)
    if RESUME:
        checkpoint=torch.load('temp/'+ args.dataset +'_best_checkpoint.pt')    
        start_epoch = checkpoint['epoch'] 
        
        scheduler.load_state_dict(checkpoint['lr_scheduler'])

    for epoch_idx in range(args.epochs+start_epoch):
        model.train()
        projection_layer.train()

        loss_train_list = []
        
        loss_train_con_list = []

        for batch_idx, (poi_seq, category_seq, hour_seq, time_seq, current_len, positive_poi, positive_category, positive_hour, positive_time, positive_current_len, one_batch_label) in enumerate(tqdm(train_dataloader)):

            poi_seq, category_seq, hour_seq, time_seq= poi_seq.to(devices[0]), category_seq.to(devices[0]), hour_seq.to(devices[0]), time_seq.to(devices[0])
            current_len, one_batch_label = current_len.to(devices[0]), one_batch_label.to(devices[0])


            train_output = model(poi_seq, category_seq, hour_seq, current_len, time_seq)
            projection_output = projection_layer(train_output)
            projection_output = F.normalize(projection_output,dim=1)

   
            positive_output  = model(positive_poi, positive_category, positive_hour, positive_current_len, positive_time)
            positive_projection_output = projection_layer(positive_output)
            positive_projection_output = F.normalize(positive_projection_output,dim=1)
            features=torch.cat([projection_output.unsqueeze(1), positive_projection_output.unsqueeze(1)], dim=1)

            loss_contrastive_loss =criterion(features, one_batch_label)

            loss_train=loss_contrastive_loss

            optimizer.zero_grad()
            loss_train.backward()
       
            optimizer.step()
            loss_train_list.append(loss_train.item())
            
            loss_train_con_list.append(loss_contrastive_loss.item())
            
        
        wandb_log.log(
                        {   
                            "train Contra-loss":np.mean(loss_train_con_list),                            
                            "train loss": np.mean(loss_train_list),
                        }
                    )
        output_content = "Train epoch:{} batch:{}   Contraloss:{:.6f}  loss:{:.6f}"               
        logger.info(output_content.format(epoch_idx, batch_idx, np.mean(loss_train_con_list),np.mean(loss_train_list)))
                
        

        model.eval()
        projection_layer.eval()
        
        gc.collect()
        torch.cuda.empty_cache()
        valid_dataloader = get_dataloader(traj_dataset = train_dataset, load_datatype='valid', batch_size=args.batch_size, sampler=valid_sampler,  user_traj_train=user_traj_train,mask_ratio=args.mask)
        loss_valid_list, y_predict_list, y_true_list, acc1_list, acc5_list = [], [], [], [], []
        loss_valid_con_list = []
            
        with torch.no_grad():

            for batch_idx, (poi_seq, category_seq, hour_seq, time_seq, current_len,positive_poi, positive_category, positive_hour, positive_time, positive_current_len,one_batch_label) in enumerate(tqdm(valid_dataloader)):

                poi_seq, category_seq, hour_seq, time_seq= poi_seq.to(devices[0]), category_seq.to(devices[0]), hour_seq.to(devices[0]), time_seq.to(devices[0])
                current_len, one_batch_label = current_len.to(devices[0]), one_batch_label.to(devices[0])
                
       

                valid_output  = model(poi_seq, category_seq, hour_seq, current_len, time_seq)
         
                val_positive_output  = model(positive_poi, positive_category, positive_hour, positive_current_len, positive_time)
                val_positive_projection_output = projection_layer(val_positive_output)
                val_positive_projection_output = F.normalize(val_positive_projection_output,dim=1)
                # -----------------------------------------------------------------------------------------------------

                val_projection_output=projection_layer(valid_output)
                val_projection_output = F.normalize(val_projection_output,dim=1)
                features=torch.cat([val_projection_output.unsqueeze(1), val_positive_projection_output.unsqueeze(1)], dim=1)
                
                loss_val_contrastive=criterion(features, one_batch_label)

                loss_valid=loss_val_contrastive


                
                loss_valid_list.append(loss_valid.item())

                loss_valid_con_list.append(loss_val_contrastive.item())

            wandb_log.log(
                    {   " Contra-epoch":epoch_idx,
                        " Contra-learnig_rate":optimizer.param_groups[0]['lr'],
                 
                        " Contra-valid loss": np.mean(loss_valid_list),
                        "alph":model.module.alph.item(),
                        
                    }
                )            
            output_content = "Valid epoch:{}   valid_loss:{:.6f} "
            logger.info(output_content.format(epoch_idx ,np.mean(loss_valid_list)))
        
        avg_train_losses.append(np.mean(loss_train_list))
        avg_valid_losses.append(np.mean(loss_valid_list))
    
        checkpoint = {
            "net": model.state_dict(),
    
            'optimizer': optimizer.state_dict(),
            "epoch": epoch_idx,
            'lr_scheduler': scheduler.state_dict()
        }

        
        user_count=0
        early_stopping(avg_valid_losses[-1], model,checkpoint)
        if early_stopping.early_stop:
            with torch.no_grad():

                for batch_idx, (poi_seq, category_seq, hour_seq, time_seq, current_len,positive_poi, positive_category, positive_hour, positive_time, positive_current_len,one_batch_label) in enumerate(tqdm(train_dataloader)):

                    poi_seq, category_seq, hour_seq, time_seq= poi_seq.to(devices[0]), category_seq.to(devices[0]), hour_seq.to(devices[0]), time_seq.to(devices[0])
                    current_len, one_batch_label = current_len.to(devices[0]), one_batch_label.to(devices[0])
                    

                    valid_output  = model(poi_seq, category_seq, hour_seq, current_len, time_seq)

                    val_projection_output = projection_layer(valid_output)
                    val_projection_output = F.normalize(val_projection_output,dim=1)
                    
                    for user_id, new_embedding in zip(one_batch_label, val_projection_output):
                        user_id=str(user_id.item())
                        if user_id not in user_embedding:
                            user_embedding[user_id] = [new_embedding]
                            user_count+=1
                        else:
                            user_embedding[user_id].append(new_embedding)
                        
                user_embedding=average_pooling(user_embedding)
                print("user_count",user_count)
            # with open('user_embedding/'+ args.dataset+ 'user_embedding.pkl', 'wb') as f:
            #         pickle.dump(user_embedding, f)
            logger.info('Early Stop!')
            break
        else:
            
            scheduler.step()
            
            
    
    return avg_train_losses, avg_valid_losses,user_embedding

def average_pooling(user_embeddings):
    avg_pooled_embeddings = {}
    for user_id, embeddings in user_embeddings.items():
   
        stacked_embeddings = torch.stack(embeddings) 
        avg_pooled_embeddings[user_id] = torch.mean(stacked_embeddings, dim=0)
    return avg_pooled_embeddings
def max_pooling(user_embeddings):
    max_pooled_embeddings = {}
    for user_id, embeddings in user_embeddings.items():
    
        stacked_embeddings = torch.stack(embeddings) 
        max_pooled_embeddings[user_id] = torch.max(stacked_embeddings, dim=0)[0]
    return max_pooled_embeddings

def min_pooling(user_embeddings):
    min_pooled_embeddings = {}
    for user_id, embeddings in user_embeddings.items():
        stacked_embeddings = torch.stack(embeddings)
        min_pooled_embeddings[user_id] = torch.min(stacked_embeddings, dim=0)[0]
    return min_pooled_embeddings
RESUME=False
def main():
    args = parse_args()
    logger = getLogger(args.dataset)
    wandb_log = wandb.init(

    project="Scale_TUL",
    name="Scale_TUL",
    notes='Scale_TUL',

    config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "dataset":args.dataset,
        "batch_size":args.batch_size,
        "Temperature":args.temperature
    },
    )
    dataset_path = './dataset/'+ args.dataset + '.csv'
    #---------------------dataset split----------------#
    user_traj_train, user_traj_test, train_nums, poi_nums, category_nums , user_nums,user_embedding = split_dataset(dataset_path)
    print("mask:",args.mask)

    #--------------get pytorch-style dataset-----------#
    train_dataset, test_dataset, train_sampler, valid_sampler = get_dataset(user_traj_train, user_traj_test, train_nums)
    devices = try_all_gpus()

    print(devices)
        

    #----------------Repeat the experiment-------------#
    for idx, seed in enumerate(random.sample(range(0, 1000), args.times)):
        
        #---------------Repeatability settings---------------#
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print(seed)
        
        #----------Building networks and optimizers----------#
        
        LSTM_timeaware_embedding = LstmTimeAwareEmbedding(args.embed_size, poi_nums, category_nums)
        LSTM_encoder = LSTMEncoder(LSTM_timeaware_embedding, args.embed_size*2)


        Temporal_encoding_layer = TemporalEncoding(args.embed_size)

        mamba_embedding=TransformerTimeAwareEmbedding(Temporal_encoding_layer,args.embed_size, poi_nums, category_nums)
        Mamba_encoder=single_Mamba(mamba_embedding,args.embed_size*2)
        model = TulNet(LSTM_encoder,Mamba_encoder,args.embed_size)
        model = nn.DataParallel(model, device_ids=devices).to(devices[0])

        projection_layer=ProjectionLayer(args.embed_size)
        projection_layer = nn.DataParallel(projection_layer, device_ids=devices).to(devices[0])
        predictor_layer=PredictorLayer(args.embed_size)
        predictor_layer = nn.DataParallel(predictor_layer, device_ids=devices).to(devices[0])
        
        optimizer = torch.optim.Adam(list(model.parameters())+list(projection_layer.parameters()), lr=args.lr)
        optimizer_predictor=torch.optim.Adam(list(predictor_layer.parameters()), lr=args.lr/2)
      
        if RESUME :
            checkpoint=torch.load('temp/'+ args.dataset +'_best_checkpoint.pt')
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])  

        # optimizer=nn.DataParallel(optimizer, device_ids=devices)
        #-------------Start training and logging-------------#
        logger.info('The {} round, start training with random seed {}'.format(idx, seed))
        current_time = time.time()
        readable_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
        # stage 1
        print('-------------stage 1 begin -------------time:',readable_time)
        user_embedding={}
      
        avg_train_losses, avg_valid_losses,user_embedding = train_model(train_dataset,train_sampler, user_embedding,valid_sampler, model,projection_layer,predictor_layer, optimizer, user_traj_train, devices, args, logger,wandb_log)
        
        # loss_with_plot(avg_train_losses, avg_valid_losses, args.dataset)
        # with open('./user_embedding.pkl', 'rb') as f:
        #     user_embedding = pickle.load(f)
        
        print('-------------stage 1 end-------------')
        logger.info("First stage time elapsed: {:.4f}s".format(time.time() - current_time))
        gc.collect()
        torch.cuda.empty_cache()
        
        print('-------------stage 2 begin-------------')
        # stage 2
        # if(len(user_embedding)==0):
        #     with open('user_embedding/'+ args.dataset +'user_embedding.pkl', 'rb') as f:
        #         user_embedding = pickle.load(f)
        #         print('pkl')
        print(len(user_embedding))
        predictor(train_dataset,train_sampler, user_embedding,valid_sampler, model,predictor_layer, optimizer_predictor, user_traj_train, devices, args, logger,wandb_log)
        print('-------------stage 2 end-------------')
        test_model(args.dataset,test_dataset, user_embedding,model,predictor_layer, devices, args, logger,wandb_log)
        logger.info("Total time elapsed: {:.4f}s".format(time.time() - current_time))
        logger.info('Fininsh trainning in seed {}\n'.format(seed))
        wandb.finish()


if __name__ == '__main__':
    main()
