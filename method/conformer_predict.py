import torch
import torch.nn as nn
from utils.model_utils import get_esm_model
from utils.data_utils import make_predict,get_predict_dataloader,centre0
from utils.evaluation_utils import threshold_predict
from model.lora import Lora_ESM
from model.conformer import Conformer
from tqdm import tqdm        
import torch
import torch.nn as nn
from tqdm import tqdm

def predict(lora_esm, tok, model, data_loader, config):
    # Set the device and evaluation mode for models
    device = config.device
    lora_esm.eval()
    model.eval()

    results = []  # Store prediction results

    with torch.no_grad():
        print('\n-------------predict------------------\n')
        # Iterate over the data loader for predictions
        for idx, (seq_list, start_list, position, id_list) in enumerate(tqdm(data_loader, desc='predict', total=len(data_loader))):
            # Process input sequences to obtain numerical representations
            _, _, num = centre0(seq_list, start_list, config.window_size, 'p', tok)
            num = num.to(device)

            # Pass inputs through the LoRA-ESM model
            esm_dict = lora_esm(num)
            hidden_representation = esm_dict['hidden']

            # Generate predictions using the main model
            output_dict = model(hidden_representation)
            outputs = config.alpha * output_dict['conv_cls'] + (1 - config.alpha) * output_dict['tran_cls']
            outputs = nn.Softmax(dim=1)(outputs)[:, 1:].view(-1)

            # Apply threshold to get binary predictions
            predict = threshold_predict(outputs, config.threshold)
            outputs = outputs.tolist()

            # Collect results for each sequence
            for i in range(len(id_list)):
                results.append({
                    "id": id_list[i],
                    "position": position[i],
                    "output": outputs[i],
                    "predict": int(predict[i])
                })

    return results


def predict_conformer(predict_ds, config):
    # Load ESM tokenizer and model
    device = config.device
    esm_tokenizer, esm_model = get_esm_model(device)

    # Load LoRA-ESM model and its checkpoint
    lora_esm = Lora_ESM(esm_model)
    lora_esm = lora_esm.to(device)
    lora_esm.load_state_dict(torch.load(config.lora_checkpoint, map_location={'cpu': device}),strict=False)

    # Load Conformer model and its checkpoint
    model = Conformer()
    model = model.to(device)
    model.load_state_dict(torch.load(config.conformer_checkpoint, map_location={'cpu': device}))

    # Prepare prediction dataset and dataloader
    dataset = make_predict(**predict_ds, windows=config.window_size)
    data_loader = get_predict_dataloader(dataset, config.batch)

    # Perform predictions
    results = predict(lora_esm, esm_tokenizer, model, data_loader, config)
    return results
