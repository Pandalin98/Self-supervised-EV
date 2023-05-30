import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Blocks import Decoder, DecoderLayer, my_Layernorm, series_decomp
from layers.Embed import cycle_Feature_Emb


def Transformer_based_deconder(d_model, nhead, output_dims, dim_feedforward=2048, batch_first=True, layer_num=2):
    decoder_layer = torch.nn.TransformerDecoderLayer(d_model, nhead=nhead, dim_feedforward=dim_feedforward)
    decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=layer_num,
                                          norm=torch.nn.BatchNorm1d(output_dims))

    return decoder


class TSDecoder_Auto(nn.Module):
    def __init__(self, input_dims, output_dims, pred_len, hidden_dims, dropout=0.1, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.c_out = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.pred_len = pred_len
        self.dropout = dropout
        self.n_heads_self = 4
        self.n_heads_cross = 8
        self.moving_avg = 25
        self.activation = F.relu
        self.layer_num = 3
        self.d_ff = 4 * self.hidden_dims
        self.decomp = series_decomp(kernel_size=self.moving_avg)
        self.dec_embedding = cycle_Feature_Emb(c_in=self.input_dims, d_model=self.hidden_dims)

        self.decoder = Decoder(
            [
                DecoderLayer(
                    # 第一个attention层，self attention
                    AutoCorrelationLayer(
                        correlation=AutoCorrelation(mask_flag=True, output_attention=False),
                        d_model=self.hidden_dims, n_heads=self.n_heads_self),
                    ##第二个attention层，用于交叉注意力
                    AutoCorrelationLayer(
                        correlation=AutoCorrelation(False, attention_dropout=self.dropout, output_attention=False),
                        d_model=self.hidden_dims, n_heads=self.n_heads_cross),
                    self.hidden_dims,
                    c_out=self.c_out,
                    d_ff=self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.layer_num)
            ],
            norm_layer=my_Layernorm(self.hidden_dims),
            projection=nn.Linear(self.hidden_dims, self.c_out, bias=True)
        )
        # self.ori_transformer = Transformer_based_deconder(d_model=self.hidden_dims, nhead=2, output_dims=1)
        # self.input_fc = nn.Linear(input_dims, hidden_dims)

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):  # x: B x T x input_dims
        # decomposition init
        mean = torch.mean(x_dec, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_dec)
        # decoder input
        trend_init = torch.cat([trend_init[:, :, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, :, :], zeros], dim=1)
        # dec
        dec_out = self.dec_embedding(seasonal_init)
        seasonal_part, trend_part = self.decoder(dec_out, x_enc, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # dec_out = self.ori_transformer(dec_out, x_enc)
        # dec_out = dec_out + trend_init

        # final
        dec_out = trend_part + seasonal_part

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == '__main__':
    x_dec = torch.Tensor(4, 1477, 1)
    x_enc = torch.Tensor(4, 1477, 64)
    input_dim = 1
    output_dim = 1
    decoder = TSDecoder_Auto(input_dim, output_dim, pred_len=800)
    print(decoder.forward(x_enc=x_enc, x_dec=x_dec).shape)
