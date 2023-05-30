import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SelfAttention_Family import AttentionLayer, FullAttention


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


class Working_Condition_Decomp(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1, output_attention_flag=False, factor=5, scale=None,
                 attention_dropout=0.1,last = False):
        super(Working_Condition_Decomp, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_attention_flag = output_attention_flag
        self.factor = factor
        self.scale = scale
        self.attention_dropout = attention_dropout
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionLayer(
            FullAttention(mask_flag=False, output_attention=self.output_attention_flag,
                          factor=self.factor, scale=self.scale, attention_dropout=self.attention_dropout),
            d_model=self.input_dim,
            n_heads=4
        )
        self.FNN = FFN(input_dim, 4 * input_dim, output_dim, 0.1)
        self.projection_w = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, bias=False)
        self.LN = nn.LayerNorm(input_dim)

    def forward(self, x_s, x_w, attn_mask=None):
        x_s = self.LN(x_s)
        x_w = self.LN(x_w)

        new_x, attn = self.attention(
            queries=x_w, keys=x_s, values=x_s,
            attn_mask=attn_mask
        )
        x_s = self.LN(x_s + self.dropout(new_x))
        x_s = self.dropout(self.FNN(x_s) + x_s)

        x_w = self.projection_w(x_w.transpose(-1, 1)).transpose(-1, 1)

        if self.output_attention_flag:
            return x_s, x_w, attn
        else:
            return x_s, x_w


class Prior_Attention_Pooler(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1, output_attention_flag=True, factor=5, scale=None,
                 attention_dropout=0.1):
        super(Prior_Attention_Pooler, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_attention_flag = output_attention_flag
        self.factor = factor
        self.scale = scale
        self.attention_dropout = attention_dropout
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionLayer(
            FullAttention(mask_flag=False, output_attention=self.output_attention_flag,
                          factor=self.factor, scale=self.scale, attention_dropout=self.attention_dropout),
            d_model=self.input_dim,
            n_heads=4
        )
        self.FNN = FFN(input_dim, 4 * input_dim, output_dim, 0.1)
        self.Prior_projection = nn.Conv1d(in_channels=1, out_channels=output_dim, kernel_size=1, bias=True)
        self.LN = nn.LayerNorm(input_dim)

    def forward(self, x, prior, attn_mask=None):
        x = self.LN(x)
        prior = self.Prior_projection(torch.reshape(prior, [-1, 1, 1])).transpose(2, 1)
        x, attn = self.attention(
            queries=prior, keys=x, values=x,
            attn_mask=attn_mask
        )

        x = self.LN(x)
        x = self.FNN(self.dropout(x))
        if self.output_attention_flag:
            return x, attn
        else:
            return x


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, math.ceil((self.kernel_size - 1) / 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) / 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Noise_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size, d_model, dropout_rate=0.1):
        super(Noise_decomp, self).__init__()
        self.moving_avg = nn.ModuleList(
            [moving_avg(kernels, stride=1) for kernels in kernel_size]
        )
        self.attention = FullAttention(mask_flag=False)
        self.out_projection = nn.Linear((len(kernel_size) + 1), 1)
        self.droupout = nn.Dropout(dropout_rate)
        self.FNN = FFN(d_model, 4 * d_model, d_model, 0.1)
        self.LN = nn.LayerNorm(d_model)
        self.BN = nn.BatchNorm1d(d_model)
        self.activation = F.relu

    def forward(self, x):
        y = torch.unsqueeze(x, 2)
        for layer in self.moving_avg:
            y = torch.cat((y, layer(x).unsqueeze(2)), 2)
        B, L, M, D = y.size()
        # y = y.view(B, L * D, M)
        normy = self.LN(y)
        y, atten = self.attention(
            normy, normy, normy,
            attn_mask=None
        )
        y = self.out_projection(y.permute(0, 1, 3, 2))
        y = y.squeeze(-1)
        x = x - y

        norm_x = self.LN(x)
        new_x = self.FNN(norm_x)
        y = x + new_x

        res = y
        moving_mean = y
        # moving_mean = self.moving_avg(x)
        # res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    ADN encoder layer with the progressive decomposition architecture
    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.LN = nn.LayerNorm(d_model)
        self.FNN = FFN(d_model, 4 * d_model, d_model, 0.1)
        self.wc_decomp = Working_Condition_Decomp(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_s, x_w, attn_mask=None):
        ## 工况分解层的残差连接
        x_s_new, x_w = self.wc_decomp(x_s, x_w)

        ## transformer block
        x_s = self.LN(x_s + x_s_new)
        new_x, attn = self.attention(
            x_s, x_s, x_s,
            attn_mask=attn_mask
        )
        x_s = self.LN(self.dropout(new_x) + x_s)

        x_s = self.dropout(self.FNN(x_s)) + x_s

        return x_s, x_w, attn


class Encoder(nn.Module):
    """
    encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, x_w, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, x_w, attn = attn_layer(x_s=x, x_w=x_w, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, x_w, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, x_w, attn = attn_layer(x, x_w, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


if __name__ == "__main__":
    x = torch.zeros([4, 10, 8])
    # # sd = Noise_decomp([2, 4, 8, 16, 32, 64])
    # # sd.forward(x)
    # WD_layer = Working_Condition_Decomp(input_dim=8, output_dim=8)
    # ND_layer = Noise_decomp([2, 4, 8, 16, 32, 64])
    # print(WD_layer(x, x))
    # print(ND_layer(x))

    encoder = EncoderLayer(
        attention=AttentionLayer(
            FullAttention(False),
            d_model=8, n_heads=4),
        d_model=8,
        d_ff=4 * 8,
        dropout=0.1)
    print(encoder(x, x))
