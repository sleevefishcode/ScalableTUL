import math
import numpy as np
from sympy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba_simple import Block
from functools import partial
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def sequence_mask(X, valid_len, value=0):
    #--------------Mask unrelated tokens in the sequence--------------#
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def MaskedMaxPooling(emb, valid_len):
    #---------------------Masked Max-pooling Layer--------------------#
    weights = torch.zeros_like(emb)
    weights = sequence_mask(weights, valid_len, value=-1e9)
    emb_pooling, _ = torch.max(emb+weights, dim=1)
    return emb_pooling
def MaskedAvgPooling(emb, valid_len):
    # ---------------------Masked Average-pooling Layer--------------------#
    # Apply sequence mask
    mask = (torch.arange(emb.size(1), device=emb.device)[None, :] < valid_len[:, None]).float()
    
    # Sum the embeddings along the sequence length dimension while ignoring padding
    emb_sum = torch.sum(emb * mask.unsqueeze(-1), dim=1)
    
    # Calculate the number of valid (non-padding) positions
    valid_len = valid_len.unsqueeze(-1).float().clamp(min=1)  # Avoid division by zero
    
    # Compute the average by dividing the sum by the number of valid positions
    emb_avg = emb_sum / valid_len
    
    return emb_avg
def calculate_similarity(label_repr_dict, user2_repr):
   
    label_repr = torch.stack(list(label_repr_dict.values()), dim=0)
    
    label_repr = F.normalize(label_repr, dim=1)

    cos = nn.CosineSimilarity(dim=-1)
    similarity_matrix =  cos(user2_repr.unsqueeze(1), label_repr.unsqueeze(0))
    top_k_indices = torch.topk(similarity_matrix, k=5, dim=1)
    # ----------------------------------------------------
    # print(top_k_indices[1])

    top_user_id = [int(list(label_repr_dict.keys())[int(indices[0])]) for indices in top_k_indices[1].tolist()]
   
    top_user_id5 = [[int(list(label_repr_dict.keys())[int(index)]) for index in indices] for indices in top_k_indices[1].tolist()]
    
    return top_user_id,top_user_id5

def calculate_and_sort_similarity(user_outputs, user2_output):


    user_matrix = torch.stack(list(user_outputs.values()), dim=0)
    

    similarities = F.cosine_similarity(user_matrix, user2_output, dim=1)

    sorted_indices = torch.argsort(similarities, descending=True)

    top_user_id = list(user_outputs.keys())[sorted_indices[0].item()]
    
    top_user_id5=[int(list(user_outputs.keys())[index.item()]) for index in sorted_indices[:5]]

    return int(top_user_id),top_user_id5


class TemporalEncoding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.w = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div = math.sqrt(1. / (embed_size*2))

    def forward(self, x, **kwargs):
        timestamp = kwargs['time_seq']  # (batch, seq_len)
        time_encode = torch.cos(timestamp.unsqueeze(-1) * self.w.reshape(1, 1, -1) + self.b.reshape(1, 1, -1))
        return self.div * time_encode


class LstmTimeAwareEmbedding(nn.Module):
    def __init__(self, embed_size, poi_nums, category_nums):
        super().__init__()
        self.embed_size = embed_size
        self.poi_embed = nn.Embedding(poi_nums+1, embed_size, padding_idx=0)
        self.category_embed = nn.Embedding(category_nums+1, embed_size, padding_idx=0)
        self.hour_embed = nn.Embedding(24+1, int(embed_size/4), padding_idx=0)
        self.fc = nn.Linear(embed_size + int(embed_size/4), embed_size)
        self.dropout = nn.Dropout(0.1)
        self.tanh = nn.Tanh()
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
      
    def forward(self, poi_seq,category_seq, hour_seq):


        poi_emb = self.poi_embed(poi_seq) 
        category_emb = self.category_embed(category_seq) 
   
        token_emb=poi_emb+category_emb 
        hour_emb = self.hour_embed(hour_seq)
     
        return self.dropout(self.tanh(self.fc(torch.cat([token_emb, hour_emb], dim=-1))))

class TransformerTimeAwareEmbedding(nn.Module):
    def __init__(self, encoding_layer, embed_size, poi_nums, category_nums):
        super().__init__()
        self.embed_size = embed_size
        self.encoding_layer = encoding_layer
        self.add_module('encoding', self.encoding_layer)
        self.poi_embed = nn.Embedding(poi_nums+1, embed_size, padding_idx=0)
        self.category_embed = nn.Embedding(category_nums+1, embed_size, padding_idx=0)
        self.hour_embed = nn.Embedding(24+1, int(embed_size/4), padding_idx=0)
        self.fc = nn.Linear(embed_size + int(embed_size/4) ,embed_size)
        self.dropout = nn.Dropout(0.1)
        self.tanh = nn.Tanh()
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
    def forward(self, poi_seq,category_seq,hour_seq , **kwargs):
        poi_emb = self.poi_embed(poi_seq) #(batch_size, seq_len, embed_size)
   
        category_emb = self.category_embed(category_seq) 

        token_emb=poi_emb+category_emb
    
        hour_emb = self.hour_embed(hour_seq)
   
        pos_embed = self.encoding_layer(poi_seq, **kwargs)
        
        return self.dropout(self.tanh(self.fc(torch.cat([token_emb, hour_emb], dim=-1)) + pos_embed))
        
class LSTMEncoder(nn.Module):
    def __init__(self, LSTM_embed, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.embed_size = LSTM_embed.embed_size
        self.LSTM_embed = LSTM_embed
        self.add_module('lstm_embed', LSTM_embed)
        
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=hidden_size, batch_first=True, num_layers=1, bidirectional=True
                            , dropout=0.1
        )
     
        self.LSTM_fc = nn.Linear(hidden_size *2, self.embed_size)
   
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)        

    def forward(self, poi_seq, category_seq, hour_seq, len_seq):

        token_emb = self.LSTM_embed(poi_seq, category_seq,hour_seq)#(batch_size, seq_len, embed_size)
  
        token_packed = pack_padded_sequence(token_emb, len_seq.cpu(), batch_first=True, enforce_sorted=False)
      
        _, (poi_hidden, _) = self.lstm(token_packed)
   
        LSTM_output = torch.cat([poi_hidden[-2, :, :], poi_hidden[-1, :, :]],dim=1)

        LSTM_output=self.LSTM_fc(LSTM_output)

 
        return LSTM_output


class single_Mamba(nn.Module):
    def  __init__(self, Mamba_embed, hidden_size, d_state=128, d_conv=2,expand=2):
        super(single_Mamba, self).__init__()
        self.embed_size = Mamba_embed.embed_size
        self.Mamba_embed = Mamba_embed
      
       
        self.add_module('Mamba_embed', Mamba_embed)
        self.encoder=Mamba(d_model=self.embed_size,  d_state=d_state, d_conv=d_conv,
                                                   expand=expand)
    def forward(self,poi_seq, category_seq, hour_seq, time_seq, len_seq):
        hidden_states  = self.Mamba_embed(poi_seq,category_seq, hour_seq
                                         , time_seq=time_seq
        )
      
        Mamba_output=self.encoder(hidden_states)

        Mamba_output=Mamba_output[:,-1,:]
        return Mamba_output
class TulNet(nn.Module):
    def __init__(self, LSTM_encoder,Mamba_encoder, user_embed_size,embed_size):
        super(TulNet, self).__init__()
        
        self.LSTM_encoder = LSTM_encoder

        self.Mamba_encoder = Mamba_encoder
        self.embed_size=embed_size
 
        self.alph = nn.Parameter(torch.tensor([0.8]), requires_grad=True)

    def forward(self, poi_seq, category_seq, hour_seq, len_seq,time_seq):
        # LSTM-encoder
        LSTM_output = self.LSTM_encoder(poi_seq, category_seq, hour_seq, len_seq)

        LSTM_output=F.normalize(LSTM_output, dim=-1)
        
        # Mamba-encoder
        Mamba_output=self.Mamba_encoder(poi_seq, category_seq,hour_seq,time_seq,len_seq)
        Mamba_output=F.normalize(Mamba_output, dim=1)
        

        
        output=self.alph*LSTM_output+(1-self.alph)*Mamba_output

        x=F.normalize(output, dim=1)
   
        return x
