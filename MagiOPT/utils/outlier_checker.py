import torch

def isOutlier(x):
    if torch.isnan(x).any():
        raise ValueError("Occur \"nan\" value, fail to iterate")
    if torch.isinf(x).any():
        raise ValueError("Occur \"inf\" value, fail to iterate")
