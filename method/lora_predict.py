import torch
import torch.nn as nn
import torch.optim as optim
from utils.model_utils import get_esm_model
from utils.data_utils import make_predict,get_predict_dataloader,centre0
from utils.evaluation_utils import threshold_predict
from model.lora import Lora_ESM
from tqdm import tqdm    
import torch
import torch.nn as nn
from tqdm import tqdm


def predict(model, tok, data_loader, config):
    # Set the device and evaluation mode for the model
    device = config.device
    model.eval()

    results = []  # Store all prediction results

    with torch.no_grad():
        print('\n-------------predict------------------\n')
        # Iterate over the data loader to generate predictions
        for idx, (seq_list, start_list, position, id_list) in enumerate(tqdm(data_loader, desc='predict', total=len(data_loader))):
            # Preprocess sequences to numerical representations
            _, _, num = centre0(seq_list, start_list, config.window_size, 'p', tok)
            num = num.to(device)

            # Pass inputs through the model
            output_dict = model(num)
            outputs = output_dict['result']
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


def predict_lora(predict_ds, config):
    # Load ESM tokenizer and model
    device = config.device
    esm_tokenizer, esm_model = get_esm_model(device)
    
    # Initialize and load LoRA-ESM model
    lora_esm = Lora_ESM(esm_model)
    lora_esm = lora_esm.to(device)
    lora_esm.load_state_dict(torch.load(config.checkpoint, map_location={'cpu': device},weights_only=True))

    # Prepare prediction dataset and dataloader
    dataset = make_predict(**predict_ds, windows=config.window_size)
    data_loader = get_predict_dataloader(dataset, config.batch)

    # Perform predictions
    results = predict(lora_esm, esm_tokenizer, data_loader, config)
    return results
