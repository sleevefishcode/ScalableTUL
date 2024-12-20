import random
from networkx import connected_watts_strogatz_graph
from sympy import true
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
class EarlyStopping_acc:


    def __init__(self, logger, dataset_name,seed, patience=3, verbose=False, delta=0):
        """[Receive optional parameters]

        Args:
            patience (int, optional): [How long to wait after last time validation loss improved.]. Defaults to 5.
            verbose (bool, optional): [If True, prints a message for each validation loss improvement. ]. Defaults to False.
            delta (int, optional): [Minimum change in the monitored quantity to qualify as an improvement.]. Defaults to 0.
        """
        self.logger = logger
        self.seed=str(seed)
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.con_start=True
        self.val_acc_max = 0.0
        # self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset_name = dataset_name
        

    def __call__(self, 
                #  val_loss, 
                val_acc,
                model,checkpoint):
        """[this is a Callback function]

        Args:
            val_loss ([float]): [The loss of receiving verification was changed to accuracy as the stop criterion in our experiment]
            model (Object): [model waiting to be saved]
        """
        # score = -val_loss
        score=val_acc
        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model,checkpoint)
            self.save_checkpoint(val_acc, model,checkpoint)
        elif score < self.best_score + self.delta:
            self.counter += 1
            con_start=False
            self.logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model,checkpoint)
            self.con_start=True
            # self.save_checkpoint(val_loss, model,checkpoint)
            self.counter = 0
    
    def save_checkpoint(self, val_acc, model,checkpoint):
        """[Saves model when validation loss decrease.]

        Args:
            val_loss ([type]): [The loss value corresponding to the best checkpoint needs to be saved]
            model (Object): [Save the model corresponding to the best checkpoint]
        """
        if self.verbose:
            self.logger.info(
                f'acc increased ({val_acc:.6f} --> {self.val_acc_max:.6f}).  Saving best model ...')
        
        torch.save(checkpoint, 'temp/'+ self.dataset_name +self.seed+ 'final_predictor_best_checkpoint.pt')
        # self.val_acc_max = val_acc
        self.val_acc_max=val_acc
class EarlyStopping_loss:
    """[Early stops the training if validation loss doesn't improve after a given patience.]
    """

    def __init__(self, logger, dataset_name,seed, patience=3, verbose=False, delta=0):
        """[Receive optional parameters]

        Args:
            patience (int, optional): [How long to wait after last time validation loss improved.]. Defaults to 6.
            verbose (bool, optional): [If True, prints a message for each validation loss improvement. ]. Defaults to False.
            delta (int, optional): [Minimum change in the monitored quantity to qualify as an improvement.]. Defaults to 0.
        """
        self.logger = logger
        self.patience = patience
        self.verbose = verbose
        self.seed=str(seed)
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.con_start=True
        # self.val_acc_max = 0.0
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset_name = dataset_name
        

    def __call__(self, 
                 val_loss, 
                # val_acc,
                model,checkpoint):
        """[this is a Callback function]

        Args:
            val_loss ([float]): [The loss of receiving verification was changed to accuracy as the stop criterion in our experiment]
            model (Object): [model waiting to be saved]
        """
        score = -val_loss
        # score=val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,checkpoint)
            # self.save_checkpoint(val_acc, model,checkpoint)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.con_start=False
            self.logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.con_start=True
            # self.save_checkpoint(val_acc, model,checkpoint)
            self.save_checkpoint(val_loss, model,checkpoint)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model,checkpoint):
        """[Saves model when validation loss decrease.]

        Args:
            val_loss ([type]): [The loss value corresponding to the best checkpoint needs to be saved]
            model (Object): [Save the model corresponding to the best checkpoint]
        """
        if self.verbose:
            self.logger.info(
                f'loss decreased ({val_loss:.6f} --> {self.val_loss_min:.6f}).  Saving best model ...')
        
        torch.save(checkpoint, 'temp/'+ self.dataset_name+self.seed + 'final_best_checkpoint.pt')
        # self.val_acc_max = val_acc
        self.val_loss_min=val_loss

def accuracy_1(pred, targ):
    """[Used to calculate trajectory links acc@1]

    Args:
        pred ([torch.tensor]): [Predicted user probability distribution]
        targ ([type]): [The real label of the user corresponding to the trajectory]

    Returns:
        [float]: [acc@1]
    """
    
    # pred = torch.max(torch.log_softmax(pred, dim=1), 1)[1]
    #ac = ((pred == targ).float()).sum().item() / targ.size()[0]
    # pred=np.array(pred)
    # print(pred)
    # print(targ)
    
    ac = ((pred == targ)).float()
 
    # print('acshape',ac.shape)
    return ac


def accuracy_5(pred, targ):
    """[Used to calculate trajectory links acc@5]

    Args:
        pred ([torch.tensor]): [Predicted user probability distribution]
        targ ([type]): [The real label of the user corresponding to the trajectory]

    Returns:
        [float]: [acc@5]
    """
    # pred = torch.topk(torch.log_softmax(pred, dim=1), k=5, dim=1, largest=True, sorted=True)[1]
    
    # ac = torch.tensor([t in pred[i].tolist() for i, t in enumerate(targ)]).float()
    ac = torch.tensor([t in p for p, t in zip(pred, targ)]).float()
    return ac


def loss_with_plot(avg_train_losses, avg_valid_losses, dataset_name):
    """[Function used to plot the loss curve and early stop line]

    Args:
        train_loss ([list]): [Loss list of training sets]
        val_loss ([list]): [Loss list of Validation sets]
    """
    # visualize the loss as the network trained
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(avg_train_losses)+1),
             avg_train_losses, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses)+1),
             avg_valid_losses, label='Validation Loss')

    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses))+1
    plt.axvline(minposs, linestyle='--', color='r',
                label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')

    # plt.ylim(0, 10) # consistent scale
    plt.xlim(0, len(avg_train_losses)+1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('log/' + dataset_name + 'early_stop_loss.png')
    

def Mask_trajectory(anchor_poi, anchor_category, anchor_time, anchor_current_len, mask_ratio):
    masked_poi = [] 
    masked_category = []  
    masked_time = []  
    masked_current_len = []  
    masked_hour = []
    for poi_seq, category_seq, time_seq, current_len in zip(anchor_poi, anchor_category, anchor_time, anchor_current_len):
    
        mask_count = int(current_len * mask_ratio)
        

        masked_indices = random.sample(range(current_len), mask_count)


        masked_poi_seq = [poi for index, poi in enumerate(poi_seq) if index not in masked_indices]
        masked_category_seq = [category for index, category in enumerate(category_seq) if index not in masked_indices]
        masked_time_seq = [timestamp for index, timestamp in enumerate(time_seq) if index not in masked_indices]


        masked_len = current_len - mask_count
        

        masked_poi.append(masked_poi_seq)
        masked_category.append(masked_category_seq)
        masked_time.append(masked_time_seq)
        masked_current_len.append(masked_len)


    max_masked_len = max(masked_current_len)


    masked_poi = [seq + [0] * (max_masked_len - len(seq)) for seq in masked_poi]
    masked_category = [seq + [0] * (max_masked_len - len(seq)) for seq in masked_category]
    masked_time = [seq + [0] * (max_masked_len - len(seq)) for seq in masked_time]
    masked_hour = [(((np.array(one_time_seq) % (24 * 60 * 60) / 60 / 60) + 8) % 24 + 1).tolist() + [0] * (max_masked_len - len(one_time_seq)) for one_time_seq in masked_time]
    

    return masked_poi, masked_category, masked_hour,masked_time, masked_current_len
