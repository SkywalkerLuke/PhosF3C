import numpy as np
import random

import yaml
import json
from typing import List, Tuple
import torch
from torch.utils.data import Dataset,DataLoader
def read_config(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        class YamlData:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def __repr__(self):
                return f"YamlData({self.__dict__})"

            def __str__(self):
                return yaml.dump(self.__dict__, default_flow_style=False)

        return YamlData(**data)



def read_data(fasta_file):
    try:
        fp = open(fasta_file)
    except IOError:
        print( 'cannot open '+fasta_file + ', check if it exist!')
        exit()
    else:
        fp = open(fasta_file)
        lines = fp.readlines()
        '''
        print("$"*100)
        print("lines :",type(lines ))
        print("lines :",lines )
        #MusiteDeep数据集带回车号
        '''
        fasta_dict = {}
        k=0
        gene_id = ""
        for line in lines:
            if line[0] == '>':   #info line
                if gene_id != "":   #if not the first, process the previous 
                    fasta_dict[gene_id] = seq 
                seq = ""
                gene_id = line[1:-1] #  line.split('|')[1] all in > need to be id            
            else:     #序列行
                seq += line.strip().replace(' ','')  
                k+=1
        fasta_dict[gene_id] = seq #last seq need to be record
        id_list=list(fasta_dict.keys())
    return fasta_dict,id_list

def read_seq(seq_list):
    fasta_dict={}
    id_list=[]
    for id,seq in enumerate(seq_list):
        id_list.append(id)
        fasta_dict[id]=seq

    return fasta_dict,id_list    

def predict_process(fasta_dict,id_list,focus):
    pre_dict={}
    all_pre=0
    for id in id_list:
        seq=fasta_dict[id]
        position_list=[]
        for index,p in enumerate(seq):
            if p in focus:
                position_list.append(index)
        pre_dict[id]=position_list
        all_pre+=len(position_list)

    print('-'*100)
    print(f'successfully process {len(id_list)} sequences')
    print("all position:",all_pre)
    print('-'*100)
    return {'fasta_dict':fasta_dict,
            'predict_dict':pre_dict,
            'id_list':id_list}


def annotation_process(fasta_dict,id_list,focus):
    p_dict={}
    all_pos=0
    del_list=[]
    for gene_id in fasta_dict:
        seq=fasta_dict[gene_id]      #str
        posnum=0; #record positive position num
        for pos in range(len(seq)):
            mid=seq[pos]          
            if(mid=='#'):
                if(posnum==0 ):
                    p_dict[gene_id]=[pos-1]           
                if(posnum!=0 ):
                    p_dict[gene_id]+=[pos-1-posnum]
                posnum+=1
        fasta_dict[gene_id]=fasta_dict[gene_id].replace('#','') #delete all #
        if posnum==0:
            p_dict[gene_id]=[]
                   
  
    for gene_id ,seq in fasta_dict.items():
        p_list=p_dict[gene_id]
        del_list=[]
        for p in p_list:
            if seq[p] in focus:
                continue      
            else:
                del_list.append(p)
        for d in del_list:        
            p_list.remove(d)       

    n_dict={}
    all_neg=0
    all_pos=0
    for id in id_list:
        n_list=[]
        p_list=p_dict[id]
        all_pos+=len(p_list)
        seq=fasta_dict[id]
        
        for index,p in enumerate(seq):
            if p in focus:
                if index in p_list:continue
                if not(index in p_list):n_list.append(index)
            else:
                continue
        all_neg+=len(n_list)
        n_dict[id]=n_list
    print('-'*100)
    print(f'successfully process {len(id_list)} sequences')
    print("positive:",all_pos)
    print("negative:",all_neg)
    print("all position:",all_neg+all_pos)
    print('-'*100)
    return {'fasta_dict':fasta_dict,
            'p_dict':p_dict,
            'n_dict':n_dict,
            'id_list':id_list}


def centre0(seq_list,start_list,windows,type='p',tok=None):
        label_content=[]
        num_content=[]
        
        
        
        for seq,start in zip(seq_list,start_list):
            #cut after padding
            data=[('id',seq)]
            seq_name, seq_strs, seq_num = tok(data)
            seq_num=seq_num[:,1:-1]
        
            #making windiw_size seq
            #pad for(pad:1; unk:3; eos:2; mask:32; cls:0))
            m_n=torch.zeros(1,windows,dtype=int)
            w_num=torch.cat([m_n,seq_num,m_n],dim=1)
            start+=windows
            

            if type=='p':
                label=[1]
                
                        
            if type=='n':
                label=[0]
        
            label_content.append(torch.tensor(label).unsqueeze(0))
            num_content.append(w_num[:,start-windows:start+windows+1])
                        
        batch_num=torch.cat(num_content,dim=0) 
        batch_single_label=torch.cat(label_content,dim=0)
        batch_bio_label=torch.cat([torch.ones(batch_single_label.size(0),1)-batch_single_label,batch_single_label],dim=-1)
                
        return batch_single_label,batch_bio_label,batch_num 
def centre1(seq_list,start_list,windows,type,tok):
        label_content=[]
        num_content=[]
        
        
        for seq,start in zip(seq_list,start_list):
            #cut after padding
            data=[('id',seq)]
            seq_name, seq_strs, seq_num = tok(data)
            seq_num=seq_num[:,1:-1]
        
            #making windiw_size seq
            #pad for(pad:1; unk:3; eos:2; mask:32; cls:0))
            m_n=torch.ones(1,windows,dtype=int)
            w_num=torch.cat([m_n,seq_num,m_n],dim=1)
            start+=windows
            

            if type=='p':
                label=[1]
                
                        
            if type=='n':
                label=[0]
        
            label_content.append(torch.tensor(label).unsqueeze(0))
            num_content.append(w_num[:,start-windows:start+windows+1])
                        
        batch_num=torch.cat(num_content,dim=0) 
        batch_single_label=torch.cat(label_content,dim=0)
        batch_bio_label=torch.cat([torch.ones(batch_single_label.size(0),1)-batch_single_label,batch_single_label],dim=-1)
        return batch_single_label,batch_bio_label,batch_num 

def centre_mix(seq_list,start_list,windows,type,tok):
        label_content=[]
        num_content=[]
        
        
        for seq,start in zip(seq_list,start_list):
            #cut after padding
            data=[('id',seq)]
            seq_name, seq_strs, seq_num = tok(data)
            seq_num=seq_num[:,1:-1]
        
            #making windiw_size seq
            #pad for(pad:1; unk:3; eos:2; mask:32; cls:0))
            l_n=torch.zeros(1,windows,dtype=int)
            r_n=torch.ones(1,windows,dtype=int)*2
            w_num=torch.cat([l_n,seq_num,r_n],dim=1)
            start+=windows

            if type=='p':
                label=[1]
                
                        
            if type=='n':
                label=[0]   

            label_content.append(torch.tensor(label).unsqueeze(0))
            num_content.append(w_num[:,start-windows:start+windows+1])
                        
        batch_num=torch.cat(num_content,dim=0) 
        batch_single_label=torch.cat(label_content,dim=0)
        batch_bio_label=torch.cat([torch.ones(batch_single_label.size(0),1)-batch_single_label,batch_single_label],dim=-1)
                
        return batch_single_label,batch_bio_label,batch_num 
def seq2emb(seq_list,tok):
    seq_list=process_list(seq_list)
    seq_name, seq_strs, seq_num = tok(seq_list)
    return seq_num[:,1:-1]
    
def process_list(list):
    return [('id',item) for item in list]




#cutting with 2*windows+1
def make_train_val_reduce(fasta_dict,p_dict,n_dict,id_list,windows,split_ratio):
    t_p_seq_list=[]
    t_n_seq_list=[]
    t_p_list=[]
    t_n_list=[]
    v_p_seq_list,v_n_seq_list=[],[]
    v_p_list,v_n_list=[],[] 
    random.shuffle(id_list)
    for id in id_list:
        n_list=n_dict[id]
        p_list=p_dict[id]
        random.shuffle(p_list)
        random.shuffle(n_list)
        n_dict[id]=n_list
        p_dict[id]=p_list
    val_num=int(len(id_list)*split_ratio)
    val_id_list=id_list[:val_num]
    train_id_list=id_list[val_num:]

    
    for i,id in enumerate(val_id_list):

        seq=fasta_dict[id].replace('\n', '')
        
        p=p_dict[id]
        n=n_dict[id]
        for i in range(len(p)):
            p_position=p[i]
            if p_position<windows:    #left padding
                p_start=p_position
            else:
                p_start=windows
            v_p_seq_list.append(seq[max(0,p_position-windows):min(len(seq),p_position+windows+1)])
            v_p_list.append(p_start)
  
            
        for i in range(len(n)):

            n_position=n[i]
            if n_position<windows:    #left padding
                n_start=n_position
            else:
                n_start=windows

            v_n_seq_list.append(seq[max(0,n_position-windows):min(len(seq),n_position+windows+1)])
            v_n_list.append(n_start)

    

    
    
    for i,id in enumerate(train_id_list):
        seq=fasta_dict[id].replace('\n', '')
        
        p=p_dict[id]
        n=n_dict[id]
        for i in range(len(p)):
            p_position=p[i]
            if p_position<windows:    #left padding
                p_start=p_position
            else:
                p_start=windows
            t_p_seq_list.append(seq[max(0,p_position-windows):min(len(seq),p_position+windows+1)])
            t_p_list.append(p_start)
        for i in range (len(n)):
            n_position=n[i]
            if n_position<windows:    #left padding
                n_start=n_position
            else:
                n_start=windows
            t_n_seq_list.append(seq[max(0,n_position-windows):min(len(seq),n_position+windows+1)])
            t_n_list.append(n_start)
    
    
    combined = list(zip(t_n_seq_list, t_n_list))

# 打乱
    random.shuffle(combined)
# 拆包回两个列表
    t_n_seq_list,t_n_list= zip(*combined)
    t_n_seq_list = t_n_seq_list[:len(t_p_seq_list)]
    t_n_list = t_n_list[:len(t_p_list)]
    return {
            't_p_seq_list' : t_p_seq_list,
            't_n_seq_list' : t_n_seq_list,
            'v_p_seq_list' : v_p_seq_list,
            'v_n_seq_list' : v_n_seq_list,
            't_p_list' : t_p_list,
            't_n_list' : t_n_list,
            'v_p_list' : v_p_list,
            'v_n_list' : v_n_list
            }

   

def make_train_val_repeat(fasta_dict,p_dict,n_dict,id_list,windows,split_ratio):
    t_p_seq_list=[]
    t_n_seq_list=[]
    t_p_list=[]
    t_n_list=[]
    v_p_seq_list=[]
    v_n_seq_list=[]
    v_p_list=[]
    v_n_list=[] 
    random.shuffle(id_list)
    for id in id_list:
        n_list=n_dict[id]
        p_list=p_dict[id]
        random.shuffle(p_list)
        random.shuffle(n_list)
        n_dict[id]=n_list
        p_dict[id]=p_list
    val_num=int(len(id_list)*split_ratio)
    val_id_list=id_list[:val_num]
    train_id_list=id_list[val_num:]

    
    for i,id in enumerate(val_id_list):

        seq=fasta_dict[id].replace('\n', '')
        
        p=p_dict[id]
        n=n_dict[id]
        for i in range(len(p)):
            p_position=p[i]
            if p_position<windows:    #left padding
                p_start=p_position
            else:
                p_start=windows
            v_p_seq_list.append(seq[max(0,p_position-windows):min(len(seq),p_position+windows+1)])
            v_p_list.append(p_start)
  
            
        for i in range(len(n)):

            n_position=n[i]
            if n_position<windows:    #left padding
                n_start=n_position
            else:
                n_start=windows

            v_n_seq_list.append(seq[max(0,n_position-windows):min(len(seq),n_position+windows+1)])
            v_n_list.append(n_start)

    

    
    
    for i,id in enumerate(train_id_list):
        seq=fasta_dict[id].replace('\n', '')
        
        p=p_dict[id]
        n=n_dict[id]
        for i in range(len(p)):
            p_position=p[i]
            if p_position<windows:    #left padding
                p_start=p_position
            else:
                p_start=windows
            t_p_seq_list.append(seq[max(0,p_position-windows):min(len(seq),p_position+windows+1)])
            t_p_list.append(p_start)
        for i in range (len(n)):
            n_position=n[i]
            if n_position<windows:    #left padding
                n_start=n_position
            else:
                n_start=windows
            t_n_seq_list.append(seq[max(0,n_position-windows):min(len(seq),n_position+windows+1)])
            t_n_list.append(n_start)
    
    
    indices = list(range(len(t_n_seq_list)))
    random.shuffle(indices)
    t_n_seq_list = [t_n_seq_list[i] for i in indices]
    t_n_list = [t_n_list[i] for i in indices]
    len_t_p_seq = len(t_p_seq_list)
    len_t_n_seq = len(t_n_seq_list)

    # 如果 t_p_seq_list 比 t_n_seq_list 短，则重复扩展 t_p_seq_list
    if len_t_p_seq < len_t_n_seq:
        repeat_times_seq = len_t_n_seq// len_t_p_seq + 1
        t_p_seq_list = (t_p_seq_list * repeat_times_seq)[:len_t_n_seq]

    # 获取 t_p_list 和 t_n_list 的长度
    len_t_p_list = len(t_p_list)
    len_t_n_list = len(t_n_list)

    # 如果 t_p_list 比 t_n_list 短，则重复扩展 t_p_list
    if len_t_p_list < len_t_n_list:
        repeat_times_list = len_t_n_list  // len_t_p_list + 1
        t_p_list = (t_p_list * repeat_times_list)[:len_t_n_list]
    return {
            't_p_seq_list' : t_p_seq_list,
            't_n_seq_list' : t_n_seq_list,
            'v_p_seq_list' : v_p_seq_list,
            'v_n_seq_list' : v_n_seq_list,
            't_p_list' : t_p_list,
            't_n_list' : t_n_list,
            'v_p_list' : v_p_list,
            'v_n_list' : v_n_list
            }

def make_test(fasta_dict,p_dict,n_dict,id_list,windows):
    t_p_seq_list=[]
    t_n_seq_list=[]
    t_p_list=[]
    t_n_list=[]
    random.shuffle(id_list)
    for id in id_list:
        n_list=n_dict[id]
        p_list=p_dict[id]
        random.shuffle(p_list)
        random.shuffle(n_list)
        n_dict[id]=n_list
        p_dict[id]=p_list
    
    for i,id in enumerate(id_list):

        seq=fasta_dict[id].replace('\n', '')
        
        p=p_dict[id]
        n=n_dict[id]
        for i in range(len(p)):
            p_position=p[i]
            if p_position<windows:    #left padding
                p_start=p_position
            else:
                p_start=windows
            t_p_seq_list.append(seq[max(0,p_position-windows):min(len(seq),p_position+windows+1)])
            t_p_list.append(p_start)
  
            
        for i in range(len(n)):

            n_position=n[i]
            if n_position<windows:    #left padding
                n_start=n_position
            else:
                n_start=windows

            t_n_seq_list.append(seq[max(0,n_position-windows):min(len(seq),n_position+windows+1)])
            t_n_list.append(n_start)

    
    return {
            'p_seq_list' : t_p_seq_list,
            'n_seq_list' : t_n_seq_list,
            'p_list' : t_p_list,
            'n_list' : t_n_list
            }

def make_predict(fasta_dict,predict_dict,id_list,windows):
    seq_list=[]
    start_list=[]
    predict_id_list=[]
    position_list=[]
    for i,id in enumerate(id_list):
        seq=fasta_dict[id].replace('\n', '')   
        p=predict_dict[id]
        for i in range(len(p)):
            position=p[i]
            if position<windows:    #left padding
                start=position
            else:
                start=windows
            seq_list.append(seq[max(0,position-windows):min(len(seq),position+windows+1)])
            start_list.append(start)
            predict_id_list.append(id)
            position_list.append(position)
  
            

    
    return {
            'seq_list' : seq_list,
            'start_list' : start_list,
            'id_list' : predict_id_list,
            'position_list' : position_list,
            }
class seq_Dataset(Dataset):
    def __init__(self, list1, list2):
        self.list1 = list1
        self.list2 = list2
       
        self.length = min(len(list1), len(list2))

    def __len__(self):
        return self.length

    def __getitem__(self, index): 
        item1 =self.list1[index]
        item2 =self.list2[index]
        return item1, item2
    def collate_fn(self, batch):

        batch_item1, batch_item2 = zip(*batch)
        batch_item1 = list(batch_item1)
        batch_item2 = list(batch_item2)

        return batch_item1, batch_item2
class id_Dataset(Dataset):
    def __init__(self, list1,list2,list3,list4):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3 
        self.list4 = list4      
        self.length = min(len(list1), len(list2))

    def __len__(self):
        return self.length

    def __getitem__(self, index): 
        item1 =self.list1[index]
        item2 =self.list2[index]
        item3 =self.list3[index]
        item4 =self.list4[index]
       
        return item1, item2, item3, item4
    def collate_fn(self, batch):
    
        batch_item1, batch_item2, batch_item3, batch_item4 = zip(*batch)
        batch_item1 = list(batch_item1)
        batch_item2 = list(batch_item2)
        batch_item1 = list(batch_item1)
        batch_item2 = list(batch_item2)

        return batch_item1, batch_item2, batch_item3, batch_item4

def get_test_dataloader(dataset,batch):
    p_test_dataset=seq_Dataset(dataset['p_seq_list'],dataset['p_list'])
    n_test_dataset=seq_Dataset(dataset['n_seq_list'],dataset['n_list'])
    p_test_dataloader = DataLoader(p_test_dataset, batch_size=batch, shuffle=False,collate_fn=p_test_dataset.collate_fn)
    n_test_dataloader = DataLoader(n_test_dataset, batch_size=batch, shuffle=False,collate_fn=n_test_dataset.collate_fn)
    return {
            "p": p_test_dataloader,
            "n": n_test_dataloader,
            }

def get_train_dataloader(dataset,batch):
    p_train_dataset=seq_Dataset(dataset['t_p_seq_list'],dataset['t_p_list'])
    p_val_dataset=seq_Dataset(dataset['v_p_seq_list'],dataset['v_p_list'])
    n_train_dataset=seq_Dataset(dataset['t_n_seq_list'],dataset['t_n_list'])
    n_val_dataset=seq_Dataset(dataset['v_n_seq_list'],dataset['v_n_list'])
    p_train_dataloader = DataLoader(p_train_dataset, batch_size=batch, shuffle=True,collate_fn=p_train_dataset.collate_fn)
    p_val_dataloader = DataLoader(p_val_dataset, batch_size=batch, shuffle=False,collate_fn=n_train_dataset.collate_fn)
    n_train_dataloader = DataLoader(n_train_dataset, batch_size=batch, shuffle=True,collate_fn=p_val_dataset.collate_fn)
    n_val_dataloader = DataLoader(n_val_dataset, batch_size=batch, shuffle=False,collate_fn=n_val_dataset.collate_fn)
    return {
    "p_t": p_train_dataloader,
    "p_v": p_val_dataloader,
    "n_t": n_train_dataloader,
    "n_v": n_val_dataloader
}

def get_predict_dataloader(dataset,batch):
    predict_dataset=id_Dataset(dataset['seq_list'],dataset['start_list'],dataset['position_list'],dataset['id_list'])
    predict_dataloader = DataLoader(predict_dataset, batch_size=batch, shuffle=False,collate_fn=predict_dataset.collate_fn)

    return predict_dataloader

