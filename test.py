from utils.data_utils import (
    read_config,
    read_data,
    annotation_process)
from method.lora_train_test import test_lora
from method.conformer_train_test import test_conformer
import argparse
import json 
import os
import torch
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",type=str,default='./hparams/conformer.yaml',help=".yaml file")
    args = parser.parse_args()

    config = read_config(args.config_path)
    test_log_path = os.path.join(config.log_path, config.log_name + '_test.json')
    data_info,idlist = read_data(config.test_path)
    test_ds = annotation_process(data_info,idlist,config.type)
    if config.model_name == 'lora':
        test_info=test_lora(test_ds,config)
    elif config.model_name == 'conformer':
        test_info=test_conformer(test_ds,config)
    os.makedirs(config.log_path, exist_ok=True)
    with open(test_log_path, 'w') as f:
        json.dump(test_info, f, indent=4)
        print(f"Test result successfully saved to: {test_log_path}")
    
    print("Testing complete.")


if __name__ == "__main__":
    main()


