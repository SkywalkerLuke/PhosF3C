import torch
import torch.nn as nn
import torch.optim as optim
from utils.model_utils import get_esm_model
from utils.data_utils import make_train_val_repeat,make_test,get_train_dataloader,get_test_dataloader,centre0
from utils.evaluation_utils import calculate_metrics,print_binary_classification_metrics,calculate_binary_classification_metrics
from model.lora import Lora_ESM
from tqdm import tqdm
import os          
import os.path

def save_state(model, config, it, metrics):
    model=model.to('cpu')
    os.makedirs(config.save_path, exist_ok=True)
    model_path = os.path.join(config.save_path, config.save_name + '.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model successfully saved to: {model_path}")
    model=model.to(config.device)
    
    acc = metrics.get('acc', 0)
    auc = metrics.get('auc', 0)
    vTP = metrics.get('vTP', 0)
    vFP = metrics.get('vFP', 0)
    vTN = metrics.get('vTN', 0)
    vFN = metrics.get('vFN', 0)
    
    p_acc = vTP / (vTP + vFN) if (vTP + vFN) > 0 else 0
    n_acc = vTN / (vTN + vFP) if (vTN + vFP) > 0 else 0
    
    state_info = {
        'step': it,
        'acc': acc,
        'auc': auc,
        'vTP': vTP,
        'vFP': vFP,
        'vTN': vTN,
        'vFN': vFN,
        'p_acc': p_acc,
        'n_acc': n_acc
    } 
    return state_info

def epoch_info(TP, TN, FP, FN,pl,ll,epoch):
    print(f'\n-------------epoch {epoch+1} info------------------\n')
    _,_=print_binary_classification_metrics(TP, TN, FP, FN,pl,ll)




def train(model,tok,data_loader,opt,criterion,config):
    device=config.device
    print('\n-------------train------------------\n')
    # Log the number of positive and negative training samples
    print(f'positive number : {len(data_loader["p_t"].dataset)}')
    print(f'positive number : {len(data_loader["n_t"].dataset)}')

    it=0
    logger=[]
    no_improvement_count=0
    best_val_acc=0
    best_val_auc=0

    # Training loop over epochs
    for epoch in tqdm(range(config.num_epoch),desc='epoch',total=config.num_epoch):
        print(f"----Epoch {epoch + 1}:-----")
        l=0
        pl=[]
        ll=[]
        TP=0
        FP=0    
        TN=0
        FN=0

        # Set LoRA-ESM to training mode
        model.train()
        for (p_train_seq_list, p_train_list), (n_train_seq_list, n_train_list) in tqdm(zip(data_loader['p_t'], data_loader['n_t']),desc='step', total=len(data_loader['p_t'])):
            opt.zero_grad()  # Reset gradients for optimizer
            it+=1  # Increment iteration count
            
            # Process positive and negative samples
            _, p_label, p_num = centre0(p_train_seq_list, p_train_list, config.window_size, 'p', tok)
            _, n_label, n_num = centre0(n_train_seq_list, n_train_list, config.window_size, 'n', tok)
            
            # Concatenate positive and negative samples
            inputs = torch.cat([n_num, p_num], dim=0)
            labels = torch.cat([n_label, p_label], dim=0)
            
            N = inputs.size(0)
            random_indices = torch.randperm(N)  # Shuffle the data
            inputs = inputs[random_indices]
            labels = labels[random_indices]
            
            # Move inputs and labels to the configured device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass through LoRA-ESM model
            output_dict = model(inputs)
            outputs=output_dict['result']

            # Calculate loss
            loss = criterion(outputs,labels)
            outputs= nn.Softmax(dim=-1)(outputs)

            # Get predicted labels and compute metrics
            _, predict = torch.max(outputs, 1)
            predict_labels = labels[:,1:].view(-1)
            l+=loss 
            tp, fp, fn, tn=calculate_metrics(predict, predict_labels)
            TP+=tp
            FP+=fp
            TN+=tn
            FN+=fn

            # Store predictions and ground truth labels
            outputs=outputs.tolist()
            labels=labels.tolist()
            pl+=outputs
            ll+=labels
            loss.backward()
            opt.step()
            if it%config.val_interval==0:
                val_info=validate(model,tok,data_loader,config,it)
                if val_info['acc'] > best_val_acc :
                    best_val_acc = val_info['acc']
                    if config.patience_key=='acc':
                        no_improvement_count = 0
                        log_info=save_state(model,config,it,val_info)
                        logger.append(log_info)
                else:
                    if config.patience_key=='acc':
                        no_improvement_count+=1

                if val_info['auc']>best_val_auc :
                    best_val_auc=val_info['auc']
                    if config.patience_key=='auc':
                        no_improvement_count = 0
                        log_info=save_state(model,config,it,val_info)
                        logger.append(log_info)
                else:
                    if config.patience_key=='auc':
                        no_improvement_count+=1

                if no_improvement_count >= config.patience_limit :
                    print(f'Early stopping! No improvement for {config.patience_limit} consecutive epochs.')
                    return logger
        epoch_info(TP, TN, FP, FN,pl,ll,epoch)



def validate(model,tok,data_loader,config,it):
    device=config.device
    print('\n-------------validate------------------\n')
    print(f'step {it}:')
    model.eval()     # Set model to evaluation mode

        
    # Initialize validation metrics
    vTP=0
    vFP=0
    vTN=0
    vFN=0
    prob_list = []  # List for predicted probabilities
    label_list = []  # List for ground truth labels

    # No gradient computation during validation
    with torch.no_grad():
        # Process positive validation samples
        for p_val_seq_list,p_val_list in data_loader['p_v']:
            _,p_val_labels,p_val_num=centre0(p_val_seq_list,p_val_list,config.window_size,'p',tok)          
            p_val_num,p_val_labels=p_val_num.to(device),p_val_labels.to(device)

            # Forward pass through LoRA-ESM for positive samples
            p_output_dict = model(p_val_num)
            p_outputs=p_output_dict['result']
            _, predict = torch.max(p_outputs, 1)
            predict_labels = p_val_labels[:,1:].view(-1)
            tp, fp, fn, tn=calculate_metrics(predict, predict_labels)
            vTP+=tp
            vFP+=fp
            vTN+=tn
            vFN+=fn

            # Store probabilities and labels for positive samples
            p_outputs= nn.Softmax(dim=1)(p_outputs)
            p_outputs=p_outputs.tolist()
            p_val_labels=p_val_labels.tolist()
            prob_list+=p_outputs
            label_list+=p_val_labels

        # Process negative validation samples           
        for n_val_seq_list,n_val_list in data_loader['n_v']:

            _,n_val_labels,n_val_num=centre0(n_val_seq_list,n_val_list,config.window_size,'n',tok)          
            n_val_num,n_val_labels=n_val_num.to(device),n_val_labels.to(device)

            # Forward pass through LoRA-ESM for negative samples
            n_output_dict = model(n_val_num)
            n_outputs=n_output_dict['result']
            _, predict = torch.max(n_outputs, 1)
            predict_labels = n_val_labels[:,1:].view(-1)
            tp, fp, fn, tn=calculate_metrics(predict, predict_labels)

            vTP+=tp
            vFP+=fp
            vTN+=tn
            vFN+=fn

            # Store probabilities and labels for negative samples
            n_outputs= nn.Softmax(dim=1)(n_outputs)
            n_outputs=n_outputs.tolist()
            n_val_labels=n_val_labels.tolist()
            prob_list+=n_outputs
            label_list+=n_val_labels
                    
        acc,auc=print_binary_classification_metrics(vTP,vFP,vTN,vFN,prob_list,label_list)
        
        return {'acc':acc,
                'auc':auc,
                'vTP':vTP,
                'vFP':vFP,
                'vTN':vTN,
                'vFN':vFN,
                'prob':prob_list,
                'label':label_list}
        
def test(model,tok,data_loader,config):
    device = config.device
    print('\n-------------test------------------\n')
    
    # Set models to evaluation mode
    model.eval()

    # Initialize variables for metrics
    vTP = 0
    vFP = 0
    vTN = 0
    vFN = 0
    prob_list = []  # To store predicted probabilities
    label_list = []  # To store true labels

    with torch.no_grad():  # No gradient computation during testing
        # Iterate over positive samples in the dataset
        for p_val_seq_list, p_val_list in tqdm(data_loader['p'], desc='positive test', total=len(data_loader['p'])):
            # Process positive samples
            _, p_val_labels, p_val_num = centre0(p_val_seq_list, p_val_list, config.window_size, 'p', tok)
            p_val_num, p_val_labels = p_val_num.to(device), p_val_labels.to(device)
            p_output_dict = model(p_val_num)
            p_outputs=p_output_dict['result']

            # Get predicted labels and compute metrics
            _, predict = torch.max(p_outputs, 1)
            predict_labels = p_val_labels[:,1:].view(-1)
            tp, fp, fn, tn=calculate_metrics(predict, predict_labels)
            # print(tp, fp, fn, tn)
            vTP+=tp
            vFP+=fp
            vTN+=tn
            vFN+=fn
            p_outputs= nn.Softmax(dim=1)(p_outputs)
            p_outputs=p_outputs.tolist()
            p_val_labels=p_val_labels.tolist()
            prob_list+=p_outputs
            label_list+=p_val_labels
            
        # Iterate over negative samples in the dataset
        for n_val_seq_list,n_val_list in tqdm(data_loader['n'],desc='negative test',total=len(data_loader['n'])):
            # Process negative samples
            _,n_val_labels,n_val_num=centre0(n_val_seq_list,n_val_list,config.window_size,'n',tok)          
            n_val_num,n_val_labels=n_val_num.to(device),n_val_labels.to(device)
            n_output_dict = model(n_val_num)
            n_outputs=n_output_dict['result']
            _, predict = torch.max(n_outputs, 1)

            # Get predicted labels and compute metrics
            predict_labels = n_val_labels[:,1:].view(-1)
            tp, fp, fn, tn=calculate_metrics(predict, predict_labels)
            vTP+=tp
            vFP+=fp
            vTN+=tn
            vFN+=fn
            n_outputs= nn.Softmax(dim=1)(n_outputs)
            n_outputs=n_outputs.tolist()
            n_val_labels=n_val_labels.tolist()
            prob_list+=n_outputs
            label_list+=n_val_labels
        print('\n-------------test result------------------\n')
                    
        # Compute and print the binary classification metrics
        _, _ = print_binary_classification_metrics(vTP, vFP, vTN, vFN, prob_list, label_list)

        # Return calculated metrics for binary classification
        return calculate_binary_classification_metrics(vTP, vFP, vTN, vFN, prob_list, label_list)
            

        
# Training wrapper
def train_lora(train_ds, config):
    device = config.device 

    # Load pre-trained ESM model and tokenizer
    esm_tokenizer, esm_model = get_esm_model(device)
    lora_esm = Lora_ESM(esm_model)
    lora_esm = lora_esm.to(device)

    # Prepare training dataset and data loader
    dataset = make_train_val_repeat(**train_ds, windows=config.window_size, split_ratio=config.split)
    data_loader = get_train_dataloader(dataset, config.batch)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    opt = optim.Adam(
        lora_esm.parameters(),  
        lr=float(config.lr), 
        betas=(config.beta1, config.beta2), 
        weight_decay=float(config.weight_decay)
    )

    # Train the model and return the log
    log = train(lora_esm, esm_tokenizer, data_loader, opt, criterion, config)
    return log

# Testing wrapper
def test_lora(test_ds, config):
    device = config.device 

    # Load the ESM model and its corresponding tokenizer
    esm_tokenizer, esm_model = get_esm_model(device)
    lora_esm = Lora_ESM(esm_model)
    lora_esm = lora_esm.to(device)

    # Load the lora checkpoint
    lora_esm.load_state_dict(torch.load(config.checkpoint, map_location={'cpu': device},weights_only=True))

    # Prepare testing dataset and data loader
    dataset = make_test(**test_ds, windows=config.window_size)
    data_loader = get_test_dataloader(dataset, config.batch)

    # Test the model and return the metrics
    test_metrics = test(lora_esm, esm_tokenizer, data_loader, config)
    return test_metrics
