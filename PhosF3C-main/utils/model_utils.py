import torch      

def get_esm_model(device):
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    batch_converter = alphabet.get_batch_converter()
    model=model.to(device)
    return batch_converter,model
