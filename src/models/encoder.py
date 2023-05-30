import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Blocks import EncoderLayer, Encoder, Working_Condition_Decomp, Prior_Attention_Pooler
from layers.Embed import cycle_Feature_Emb, TokenEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.dilated_conv import DilatedConvEncoder


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.05):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


def Transformer_based_enconder(d_model, nhead, output_dims, dim_feedforward, batch_first=True, layer_num=1):
    encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead=nhead, dim_feedforward=dim_feedforward)
    encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layer_num)

    squential = nn.Sequential(encoder
                              , nn.Linear(d_model, output_dims))
    return squential


class ADN(nn.Module):
    def __init__(self, input_dims, wc_input_dim, output_dims, hidden_dims,args, mask_mode='continuous',
                 n_heads=4, moving_avg=25, dropout_rate=0.1, activation=F.relu, feature_extractor_layer=2,
                 ):
        super().__init__()
        self.input_dims = input_dims
        self.wc_input_dim = wc_input_dim
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.n_heads = n_heads
        self.moving_avg = moving_avg
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.feature_extractor_layer = feature_extractor_layer
        self.input_projection = cycle_Feature_Emb(c_in=self.input_dims, d_model=self.hidden_dims)
        self.wc_projection = cycle_Feature_Emb(c_in=self.wc_input_dim, d_model=self.hidden_dims)
        # self.output_attention = AttentionLayer(
        #     FullAttention(False, output_attention=True),
        #     d_model=self.hidden_dims, n_heads=2)
        self.Prior_Attention_Pooler = Prior_Attention_Pooler(input_dim=self.hidden_dims, output_dim=self.hidden_dims)
        self.output_projection = nn.Linear(self.hidden_dims, self.output_dims)

        self.feature_extractor = Encoder([EncoderLayer(
            attention=AttentionLayer(
                FullAttention(False),
                d_model=self.hidden_dims, n_heads=4),
            d_model=self.hidden_dims,
            d_ff=self.hidden_dims,
            dropout=self.dropout_rate,
            activation=self.activation)
            for l in range(self.feature_extractor_layer)
        ],
            norm_layer=nn.LayerNorm(self.hidden_dims)
        )

        last_layer_number = self.feature_extractor_layer - 1
        self.feature_extractor.attn_layers[last_layer_number].wc_decomp.projection_w.weight.detach()
        self.feature_extractor.attn_layers[last_layer_number].wc_decomp.projection_w.weight.requires_grad = False
        # # 创建一个与原始weight具有相同形状和数据类型的新Tensor
        # original_weight = self.feature_extractor.attn_layers[last_layer_number].wc_decomp.projection_w.weight
        # new_weight = torch.randn_like(original_weight)

        # # 将新Tensor分配给weight属性
        # self.feature_extractor.attn_layers[last_layer_number].wc_decomp.projection_w.weight = new_weight



    def forward(self, x, x_w, prior, mask=None):  
        
        #把nan值替换为0
        x = x.masked_fill(~x.isnan(), 0)
        x_w = x_w.masked_fill(~x_w.isnan(), 0)
        x_w = self.wc_projection(x_w)# B x T x Ch
        x = self.input_projection(x)  # B x T x Ch
        x,_= self.feature_extractor(x, x_w)
        x = torch.mean(x, dim=1)
        x = self.output_projection(x)
        # prior = prior/1000
        # x, _ = self.Prior_Attention_Pooler(x, prior)
        # x = torch.avg_pool1d(x.transpose(1, 2), kernel_size=x.size(1)).squeeze(-1)
        # x = self.output_projection(x[:, 0, :])

        return x


class TCN_Encoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x


if __name__ == '__main__':
    x = torch.zeros(4, 400, 3)
    x_w = torch.zeros(4, 400, 4)
    input_dim = 3
    output_dim = 128
    hidden_dims = 512
    wc_input_dim = 4
    encoder = ADN(input_dim, wc_input_dim, output_dim, hidden_dims)
    print(encoder.forward(x, x_w).shape)
