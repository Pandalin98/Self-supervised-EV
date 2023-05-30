import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .AutoCorrelation import AutoCorrelationLayer, AutoCorrelation
from .Embed import TokenEmbedding


class Working_Condition_Decomp(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1, output_attention_flag=False, factor=1, scale=None,
                 attention_dropout=0.1):
        super(Working_Condition_Decomp, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_attention_flag = output_attention_flag
        self.factor = factor
        self.scale = scale
        self.attention_dropout = attention_dropout
        self.dropout = nn.Dropout(dropout)
        self.attention = AutoCorrelationLayer(
            AutoCorrelation(mask_flag=False, output_attention=self.output_attention_flag,
                            factor=self.factor, scale=self.scale, attention_dropout=self.attention_dropout),
            d_model=self.input_dim,
            n_heads=4
        )
        self.projection = TokenEmbedding(self.input_dim, self.output_dim)

    def forward(self, x_s, x_w, attn_mask=None):
        new_x, attn = self.attention(
            queries=x_w, keys=x_s, values=x_s,
            attn_mask=attn_mask
        )
        x_s = x_s + self.dropout(new_x)
        x_w = self.projection(x_w)
        if self.output_attention_flag:
            return x_s, x_w
        else:
            return x_s, x_w, attn


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


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

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = nn.ModuleList(
            [moving_avg(kernels, stride=1) for kernels in kernel_size]
        )
        self.attention = AutoCorrelation()
        self.out_projection = nn.Linear((len(kernel_size) + 1), 1)

    def forward(self, x):
        y = torch.unsqueeze(x, 2)
        for layer in self.moving_avg:
            y = torch.cat((y, layer(x).unsqueeze(2)), 2)
        B, L, M, D = y.size()
        # y = y.view(B, L * D, M)
        y = self.attention(
            y, y, y,
            attn_mask=False
        )[0]

        y = self.out_projection(y.permute(0, 1, 3, 2))
        y = y.squeeze(-1)
        res = x - y
        moving_mean = y
        # moving_mean = self.moving_avg(x)
        # res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # self.wcdecomp1 = Working_Condition_Decomp(d_model, d_model)
        # self.wcdecomp2 = Working_Condition_Decomp(d_model, d_model)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.wc_decomp1 = Working_Condition_Decomp(d_model, d_model)
        self.wc_decomp2 = Working_Condition_Decomp(d_model, d_model)
        self.wc_decomp3 = Working_Condition_Decomp(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_s, x_w, attn_mask=None):
        # x_s, x_w = self.wcdecomp1(x_s=x_s, x_w=x_w)

        x_s, x_w, _ = self.wc_decomp1(x_s, x_w)
        new_x, attn = self.attention(
            x_s, x_s, x_s,
            attn_mask=attn_mask
        )
        x_s = x_s + self.dropout(new_x)
        x_s, _ = self.decomp1(x_s)
        x_s, x_w_new, _ = self.wc_decomp2(x_s, x_w)
        # x_w, _ = self.decomp1(x_w)
        # x_s, x_w = self.wcdecomp2(x_s=x_s, x_w=x_w)
        y = x_s
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x_s + y)
        return res, x_w_new, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
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


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """

    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp0 = series_decomp(moving_avg)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        # x = x + self.dropout(self.cross_attention(
        #     x, cross, cross,
        #     attn_mask=cross_mask
        # )[0])
        # x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend3
        # residual_trend =trend1+trend0
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, trend


if __name__ == "__main__":
    x = torch.zeros([4, 300, 3])
    sd = Noise_decomp([2, 4, 8, 16, 32, 64])
    sd.forward(x)
