from utils.data_utils import (
    read_config,
    read_fasta,
    predict_process)
from method.lora_predict import predict_lora
from method.conformer_predict import predict_conformer
import argparse
import os         
import json
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",type=str,default='./hparams/conformer.yaml',help=".yaml file")
    args = parser.parse_args()

    config = read_config(args.config_path)  
    predict_result_path = os.path.join(config.predict_result_path, config.save_name + '.json')
    data_info,idlist= read_fasta(config.predict_path)
    predict_ds = predict_process(data_info,idlist,config.type)
    if config.model_name == 'lora':
        result=predict_lora(predict_ds,config)
    elif config.model_name == 'conformer':
        result=predict_conformer(predict_ds,config)
    os.makedirs(config.predict_result_path, exist_ok=True)
    with open(predict_result_path, 'w') as f:
        json.dump(result, f, indent=4)
        print(f"Predict result successfully saved to: {predict_result_path}")
    
    print("Predicting complete.")


if __name__ == "__main__":
    main()

