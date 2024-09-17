import torch

def default_preprocess_obss(obss, device=None):
    if obss.ndim>1:
        return torch.tensor(obss, device=device, dtype = torch.float32).reshape(obss.shape[0],1,8,11)
    else:
        return torch.tensor(obss, device=device, dtype = torch.float32).reshape(1,1,8,11)