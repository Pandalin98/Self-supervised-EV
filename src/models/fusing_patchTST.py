
__all__ = ['Fusing_PatchTST']
import sys
sys.path.append('/data/home/ps/WYL/power_battery')
# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F

import numpy as np

from collections import OrderedDict
from src.models.layers.pos_encoding import *
from src.models.layers.basics import *
from src.models.layers.attention import *
from src.models.layers.Embed import Patch_Emb,Decoder_Emb,TemporalEmbedding
from src.models.layers.SelfAttention_Family import AttentionLayer, FullAttention

# Cell
##Define the symbol 
## batch size : bs
#length of the cycle : cl
#lenght of time series : tl
#length of the patch : pl
#number of patch : np
#length of the feature : fl

class Fusing_PatchTST(nn.Module):
    """
    Output dimension: 
         [bs x tl x fl] for prediction
         [bs x fl] for regression
         [bs x fl] for classification
         [bs xcl x np x fl*pl] for pretrain
    """
        ##对head进行修改，修改输入输出的dim
    def __init__(self, c_in:int , patch_len:int, stride:int,c_in_dec=1 ,  target_dim =1,
                 n_layers:int=3, n_layers_dec=1,d_model=128, n_heads=16, n_heads_dec = 8,shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "regression", individual = False, 
                 y_range:Optional[tuple]=None, verbose:bool=False,prior_dim =1,output_attention = False, **kwargs):

        super().__init__()
        self.y_range = y_range
        assert head_type in ['pretrain', 'prior_pooler', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'
        # Backbone
        self.backbone = PatchTSTEncoder(c_in, patch_len=patch_len, stride=stride,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        # todo 后期需要创一个带日期的embedding
    
        self.dec_emb = Decoder_Emb(c_in_dec,d_model)
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads_dec,
                dim_feedforward=d_ff,
                dropout=dropout,
            )
            ,num_layers=n_layers_dec,
            norm=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2)) if norm=='BatchNorm'
            else nn.LayerNorm(d_model))
        self.output_attention = output_attention
        self.TemporalEmbedding = TemporalEmbedding(d_model=d_model)
        # Head
        self.n_vars = c_in
        self.head_type = head_type
        self.d_model = d_model
        self.prior_dim = prior_dim
        ##todo 后期修改prior dim的输入
        self.prior_projection  = nn.Linear(self.prior_dim, d_model)
        if head_type == "pretrain":
            self.head_re = PretrainHead_regression(d_model, patch_len, self.n_vars,head_dropout) # custom head passed as a partial func with all its kwargs
            self.head_cons1 = AttentionAggregation(d_model)
            self.head_cons1_1 = PretrainHead_constrat(d_model)
            self.head_cons2 = PretrainHead_constrat(d_model)
        elif head_type == "prior_pooler":
            self.head = Prior_Attention_Pooler(d_model, d_ff, d_model, head_dropout,output_attention_flag = output_attention)
        elif head_type == "regression":
            self.head = RegressionHead(5, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)
        self.out_put_projection = nn.Linear(d_model, 1)

    def forward(self, enc_in,prior,dec_in,enc_mark):                             
        """
        input:
        enc_in: tensor [bs x cl x np x fl * pl]
        enc_mark tensor [bs x cl x 4]

        prior: tensor [bs x cl x prior_dim]
        output:
        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs * cl x np x  fl * pl]  and [bs*cl x np x d_model ]for pretrain
        """   
        bs,cl,np,fpl = enc_in.shape
        enc_in = enc_in.view(bs*cl,np,fpl)
        if self.head_type == "pretrain":    
            prior = self.prior_projection(prior)
            z_enc = self.backbone(enc_in)                                                                # z: [bs x nvars x d_model x num_patch]
            con1 = self.head_cons1_1(self.head_cons1(z_enc))
            con2 = self.head_cons2(prior.view(bs*cl,self.d_model))
            ##con2 复制cl次

            re = self.head_re(z_enc).view(bs,cl,np,-1)     
            return re,con1,con2 
        else:
            z_enc = self.backbone(enc_in) #[bs*cl, np, d_model]
            if self.head_type == "prior_pooler":
                prior = prior.view(bs*cl,self.prior_dim)
                prior = self.prior_projection(prior)
                z_enc_out = self.head(z_enc,prior)
            else:
                z_enc_out = self.head(z_enc)
            
            z_enc_out=z_enc_out.view(bs,cl,self.d_model)  
            z_enc_out = z_enc_out+self.TemporalEmbedding(enc_mark)
            dec_in = self.TemporalEmbedding(dec_in[:,:,:4])+self.dec_emb(dec_in[:,:,4:])
            output = self.decoder( dec_in.transpose(0,1),z_enc_out.transpose(0,1)).transpose(0,1)
            y = self.out_put_projection(output)         # y: bs x output_dim
            if self.y_range: y = SigmoidRange(*self.y_range)(y)        
            return y.squeeze(-1)

class FFN(nn.Module):
    """基于位置的前馈网络"""

    def __init__(self, ffn_num, ffn_num_hiddens, ffn_out, dropout_rate=0.1):
        super(FFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_out)

    def forward(self, X):
        return self.dense2(self.dropout(self.relu(self.dense1(X))))
    
class Prior_Attention_Pooler(nn.Module):
    def __init__(self, d_model,d_ff, target_dim, dropout=0, output_attention_flag=False, factor=5, scale=None,
                 attention_dropout=0):
        super(Prior_Attention_Pooler, self).__init__()
        self.input_dim = d_model
        self.output_dim = target_dim
        self.output_attention_flag = output_attention_flag
        self.factor = factor
        self.scale = scale
        self.attention_dropout = attention_dropout
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionLayer(
            FullAttention(mask_flag=False, output_attention=True,
                          factor=self.factor, scale=self.scale, attention_dropout=self.attention_dropout),
            d_model=self.input_dim,
            n_heads=4
        )
        self.FNN = FFN(d_model, d_ff, d_model, dropout)
        self.LN = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, target_dim)
        self.y_range = [0,1]   
    def forward(self, x, prior, attn_mask=None):
        x = self.LN(x)
        #增加一个维度
        prior = prior.unsqueeze(1)
        x, attn = self.attention(
            queries=prior, keys=x, values=x,
            attn_mask=attn_mask
        )

        x = self.LN(x)
        x = self.dropout(self.FNN(x))+x
        x = x.squeeze(1)
        x = self.linear(x)
        if self.y_range: x = SigmoidRange(*self.y_range)(x)
        if self.output_attention_flag:
            return x, attn
        else:
            return x


class RegressionHead(nn.Module):
    def __init__(self, n_samples, d_model, output_dim, head_dropout, y_range=None,pooler_type="lastN"):
        super().__init__()
        self.y_range = [0,1]
        self.n_samples = n_samples
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_samples*d_model, output_dim)  
        if pooler_type == "lastN":
            self.linear = nn.Linear(n_samples*d_model, output_dim)
        else:
            self.linear = nn.Linear(d_model, output_dim)
        self.pooler_type = pooler_type
    def forward(self, x):
        """
        x: [bs x num_patch x d_model ]
        output: [bs x output_dim]
        """
        if self.pooler_type == "lastN":
            x = x[:,-self.n_samples:,:]    # only consider the last item in the sequence, x: bs x nvars x d_model
        elif self.pooler_type == "mean":
            x = torch.avg_pool1d(x.transpose(1, 2), kernel_size=x.shape[1]).squeeze()   # x: bs x d_model
        elif self.pooler_type == "max":
            x = F.max_pool1d(x.transpose(1, 2), kernel_size=x.shape[1]).squeeze()          # bs x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)        
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_samples, d_model, n_classes, head_dropout,pooler_type="lastN"):
        super().__init__()
        self.n_samples = n_samples
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.pooler_type = pooler_type
        if pooler_type == "lastN":
            self.linear = nn.Linear(n_samples*d_model, n_classes)
        else:
            self.linear = nn.Linear(d_model, n_classes)
    def forward(self, x):
        """
        x: [bs x num_patch x d_model ]
        output: [bs x n_classes]
        """
        if self.pooler_type == "lastN":
            x = x[:,-self.n_samples,:]    # only consider the last item in the sequence, x: bs x nvars x d_model
        elif self.pooler_type == "mean":
            x = torch.mean(x, dim=1)
        elif self.pooler_type == "max":
            x = torch.max(x, dim=1)[0]
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class PretrainHead_regression(nn.Module):
    def __init__(self, d_model, patch_len,nvars, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(d_model,  patch_len*nvars))
    def forward(self, x):
        """
        x: tensor [bs   x num_patchx d_model]
        output: tensor [bs x num_patch x patch_len]
        """
        x = self.projection( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
                                                 # [bs x num_patch x nvars x patch_len]
        return x


class AttentionAggregation(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.W_Q = nn.Linear(input_dim, input_dim)
        self.W_K = nn.Linear(input_dim, input_dim)
        self.W_V = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # x: (batch_size, patch_len, input_dim)
        Q = self.W_Q(x)  # (batch_size, patch_len, input_dim)
        K = self.W_K(x)  # (batch_size, patch_len, input_dim)
        V = self.W_V(x)  # (batch_size, patch_len, input_dim)

        # Calculate attention scores
        scores = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, patch_len, patch_len)
        attention = nn.functional.softmax(scores, dim=-1)  # (batch_size, patch_len, patch_len)

        # Apply attention to values
        out = torch.bmm(attention, V)  # (batch_size, patch_len, input_dim)
        out = out.mean(dim=1)  # (batch_size, input_dim)
        return out


class PretrainHead_constrat(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model))
    def forward(self, x):
        """
        x: tensor [bs   x num_patchx d_model]
        output: tensor [bs x d_model]
        """
        #判断是否是3d的

        if len(x.shape) == 3:
            x = torch.avg_pool1d(x.transpose(1, 2), kernel_size=x.shape[1]).squeeze()        
        x = self.projection(x)      # [bs x nvars x num_patch x patch_len]
                                                 # [bs x num_patch x nvars x patch_len]
        return x

class PatchTSTEncoder(nn.Module):

    def __init__(self, c_in, patch_len,stride,
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        ##主要改动的地方，改成mixing的情况
        self.n_vars = c_in
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding        

        # Input encoding: projection of feature vectors onto a d-dim vector space
        self.patch_embed = Patch_Emb(c_in, d_model, patch_len,stride)

        # if not shared_embedding: 
        #     self.W_P = nn.ModuleList()
        #     for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        # else:
        #     self.W_P = nn.Linear(patch_len, d_model)      



        # Positional encoding

        # Residual dropout
        self.dropout = nn.Dropout(dropout)


        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn)

    def forward(self, x) -> Tensor:          
        """
        x: tensor [bs xseries_len x nvars]
        """
        # Input encoding
        u =self.patch_embed(x)                                                # u:  [bs x cl x np  x d_model]
        ## encoding
        z = self.encoder(u)                                                   # z: [bs x  cl x np  x d_model]
        return z
    
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:


            return src




if __name__ == '__main__':
    
    x = torch.zeros(3,4, 3, 400) #bs x cl x np x fl*pl
    prior = torch.zeros(3,4,1) 
    x_dec = torch.ones(3,6,5)
    x_mark = torch.ones(3,4,4)
    model = Fusing_PatchTST(c_in=4,c_in_dec =1,
                target_dim=1,
                patch_len=100,
                stride=12,
                n_layers=4,
                n_heads=16,
                d_model=128,
                shared_embedding=True,
                d_ff=256,                        
                dropout=0.1,
                head_dropout=0.1,
                act='relu',
                head_type='prior_pooler',
                res_attention=False
                )        
    print(model.forward(x,prior,x_dec,x_mark).shape)
