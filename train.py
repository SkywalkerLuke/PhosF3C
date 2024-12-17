from utils.data_utils import (
    read_config,
    read_data,
    annotation_process)
from method.lora_train_test import train_lora
from method.conformer_train_test import train_conformer
import argparse
import os
import json
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",default='./hparams/conformer.yaml',type=str,help=".yaml file")
    args = parser.parse_args()

    config = read_config(args.config_path)
    log_path = os.path.join(config.log_path, config.log_name + '_save.json')
    data_info,id_list = read_data(config.train_path)
    train_ds = annotation_process(data_info,id_list,config.type)
    if config.model_name == 'lora':
        log=train_lora(train_ds,config)
    elif config.model_name == 'conformer':
        log=train_conformer(train_ds,config)
    os.makedirs(config.log_path, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=4)
        print(f"Log successfully saved to: {log_path}")
    
    print("Training complete.")

if __name__ == "__main__":
    main()
