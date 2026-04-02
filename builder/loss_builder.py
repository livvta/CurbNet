

import torch
from utils.lovasz_losses import lovasz_softmax, FocalLoss


# def build(wce=True, lovasz=True, num_class=20, ignore_label=0):

#     loss_funs = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

#     if wce and lovasz:
#         return loss_funs, lovasz_softmax
#     elif wce and not lovasz:
#         return wce
#     elif not wce and lovasz:
#         return lovasz_softmax
#     else:
#         raise NotImplementedError


### when use improved focal loss 
def build(alpha, gamma=2, wce=True, lovasz=True, num_class=20, ignore_label=0):
    if wce:
        loss_funs = FocalLoss(alpha=alpha , gamma=2, ignore_index=ignore_label)  # improved focal loss 
        # loss_funs = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)   # origin focal loss 
    # else:
    #     loss_funs = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    if wce and lovasz:
        return loss_funs, lovasz_softmax
    elif wce and not lovasz:
        return loss_funs
    elif not wce and lovasz:
        return lovasz_softmax
    else:
        raise NotImplementedError