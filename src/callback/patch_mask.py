
import torch
from torch import nn

from .core import Callback

# Cell
class PatchCB(Callback):

    def __init__(self, patch_len, stride ):
        """
        Callback used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self): self.set_patch()
       
    def set_patch(self):
        """
        take xb from learner and convert to patch: [bs x seq_len × nvars] -> [bs x num_patch x patch_len*nvars]
        """
        bs,cl,tl,fl = self.xb.shape
        xb = self.xb.reshape(bs*cl,tl,fl)
        xb_patch, num_patch = create_patch(xb, self.patch_len, self.stride)    # xb: [bs x seq_len × nvars ]
        self.learner.xb = xb_patch.reshape(bs,cl,num_patch,-1)                              # xb_patch: [bs x cl x num_patch x patch_len*nvars]        


class PatchMaskCB(Callback):
    def __init__(self, patch_len, stride, mask_ratio,
                        mask_when_pred:bool=False):
        """
        Callback used to perform the pretext task of reconstruct the original data after a binary mask has been applied.
        Args:
            patch_len:        patch length
            stride:           stride
            mask_ratio:       mask ratio
        """
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio        

    def before_fit(self):
        device = self.learner.device       
 
    def before_forward(self): self.patch_masking()
        
    def patch_masking(self):
        """
        xb: [bs x seq_len x n_vars] -> [bs x num_patch x patch_len]
        """
        bs,cl,tl,fl = self.xb.shape
        xb = self.xb.reshape(bs*cl,tl,fl)
        xb_patch, num_patch = create_patch(xb, self.patch_len, self.stride)    # xb_patch: [bs x num_patch x patch_len]
        xb_mask, _, self.mask, _ = random_masking_3D(xb_patch, self.mask_ratio)   # xb_mask: [bs x num_patch  x patch_len]
        self.learner.mask = self.mask.reshape(bs,cl,num_patch).bool()    # mask: [bs×cl x num_patch ]
        self.learner.xb = xb_mask.reshape(bs,cl,num_patch,-1)       # learner.xb: masked 4D tensor    
        self.learner.target = xb_patch.reshape(bs,cl,num_patch,-1)    # learner.target: non-masked 4D tensor
 


def create_patch(input_tensor, patch_size, stride):
    """
    input_tensor: [bs x seq_len x n_vars]
    return patch: [bs x num_patch x patch_len*n_vars]
    """
    batch_size, seq_len, n_vars = input_tensor.shape

    # 计算需要补齐的数量
    num_patches = (seq_len - patch_size) // stride + 1
    padding = (stride - seq_len % stride) % stride
    pad_num = (patch_size - (seq_len + padding) % patch_size) % patch_size
    padding += pad_num
    if padding > 0:
        input_tensor = torch.cat([input_tensor, input_tensor[:, -1:, :].repeat(1, padding, 1)], dim=1)
    seq_len_padded = seq_len + padding
    patch_num = (seq_len_padded - patch_size) // stride + 1
    patch = input_tensor.unfold(1, patch_size, stride).reshape(batch_size, patch_num, -1)
    patch = patch.reshape(batch_size, patch_num, -1)
    return patch, patch_num




def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, nvars,device=xb.device)  # noise in [0, 1], bs x L x nvars
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]                                              # ids_keep: [bs x len_keep x nvars]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x len_keep x nvars  x patch_len]
   
    # removed x
    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)                 # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)                                  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore


def random_masking_3D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]                                                 # ids_keep: [bs x len_keep]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))        # x_kept: [bs x len_keep x dim]
   
    # removed x
    x_removed = torch.zeros(bs, L-len_keep, D, device=xb.device)                        # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D))    # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)                                          # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch]
    return x_masked, x_kept, mask, ids_restore


if __name__ == "__main__":
    bs, L, nvars, D = 2,80,5
    xb = torch.randn(bs, L, nvars, D)
    xb_mask, mask, ids_restore = random_masking_3D(xb, mask_ratio=0.5)
    breakpoint()


