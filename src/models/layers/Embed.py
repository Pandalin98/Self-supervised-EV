import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class LearnedPositionEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super(LearnedPositionEncoding, self).__init__()
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        position_embeddings = self.position_embedding(positions)
        return self.dropout(position_embeddings)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    

class PatchTokenEmbedding(nn.Module):
        def __init__(self, c_in, d_model,patch_len=12,stride =12):
            super(PatchTokenEmbedding, self).__init__()
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.patchify = Patchify(patch_len,stride)
            self.tokenConv = nn.Conv1d(in_channels=c_in*patch_len, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular', bias=False)
            self.stride = stride
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

        def forward(self, x):

            x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
            return x

class Patchify:
    def __init__(self, patch_size, stride):
        self.patch_size = patch_size
        self.stride = stride

    def __call__(self, input_tensor):
        batch_size, seq_len, _ = input_tensor.shape
        padding = (self.stride - seq_len % self.stride) % self.stride
        if padding > 0:
            input_tensor = torch.cat([input_tensor, input_tensor[:, -1:, :].repeat(1, padding, 1)], dim=1)
        patch_num = (seq_len + padding) // self.stride
        input_tensor = input_tensor.unfold(1, self.patch_size, self.stride).reshape(batch_size, patch_num, -1)
        return input_tensor


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class cycle_Feature_Emb(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(cycle_Feature_Emb, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = LearnedPositionEncoding(c_in)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x + self.position_embedding(x))
        return self.dropout(x)


class Patch_Emb(nn.Module):
    def __init__(self, c_in, d_model,patch_len = 12,stride =12):
        super(Patch_Emb, self).__init__()
        self.value_embedding = PatchTokenEmbedding(c_in=c_in, d_model=d_model,patch_len =patch_len,stride =stride)
        self.position_embedding = LearnedPositionEncoding(d_model)

    def forward(self, x):
        #  x: tensor [bs xseries_len x nvars]

        x = self.value_embedding(x)
        x= x+self.position_embedding(x)

        # x:tensor[bs x n_patches x d_model]
        return x
    
if __name__ == '__main__':
    
    x = torch.zeros(4, 400, 13)
    input_dim = 13
    d_model = 128
    emb = Patch_Emb(input_dim, d_model)
    print( str(emb(x).shape))
