import torch      
import torch.nn as nn

class esm_model(nn.Module):
    def __init__(self,model):
        super(esm_model,self).__init__()
        self.model=model
    def forward(self,**kwargs):
        return self.model(tokens=kwargs['tokens'],repr_layers=kwargs['repr_layers'])
    
def get_esm_model(device):
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    batch_converter = alphabet.get_batch_converter()
    model = esm_model(model)
    model=model.to(device)
    return batch_converter,model



