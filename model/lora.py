import torch.nn as nn
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

class Lora_ESM(nn.Module):
    def __init__(self,model):
        super(Lora_ESM,self).__init__()
        self.peft_config = LoraConfig(
        target_modules=['q_proj', 'out_proj', 'v_proj', 'k_proj','regression','dense'],
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.2)
        self.lora_model = get_peft_model(model, self.peft_config)
        self.lora_model.print_trainable_parameters()
        self.proj1=nn.Linear(1280,1)
        self.proj2=nn.Linear(31,2)
    def forward(self,x):
        result=self.lora_model(tokens=x, repr_layers=[33])
        last_hidden=result['representations'][33]
        logits=self.proj1(last_hidden).squeeze(-1)
        result=self.proj2(logits)
        
        
        return {'result':result,
                'hidden':last_hidden}
    
class Lora_ESM_large(nn.Module):
    def __init__(self,model):
        super(Lora_ESM_large,self).__init__()
        self.peft_config = LoraConfig(
        target_modules=['q_proj', 'out_proj', 'v_proj', 'k_proj','regression','dense'],
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.2)
        self.lora_model = get_peft_model(model, self.peft_config)
        self.lora_model.print_trainable_parameters()
        self.proj11=nn.Linear(2560,1280)
        self.proj12=nn.Linear(1280,1)
        self.proj2=nn.Linear(31,2)
    def forward(self,x):
        result=self.lora_model(tokens=x, repr_layers=[33])
        last_hidden=result['representations'][33]
        last_hidden=self.proj11(last_hidden)
        logits=self.proj12(last_hidden).squeeze(-1)
        result=self.proj2(logits)
        
        
        return {'result':result,
                'hidden':last_hidden}
class Lora_Raw(nn.Module):
    def __init__(self,model):
        super(Lora_Raw,self).__init__()
        self.lora_model=model
    def forward(self,x):
        result=self.lora_model(x, repr_layers=[33], return_contacts=False)
        last_hidden=result['representations'][33]

        return {'result':0,
                'hidden':last_hidden}  

