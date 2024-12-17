from sklearn.metrics import roc_auc_score
import numpy as np
import torch
def print_binary_classification_metrics(tp, fp, tn, fn,prob,label):
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    auc = roc_auc_score(label, prob)
    p_acc=tp/(tp+fn) if (tp+fn) >0 else 0
    n_acc=tn/(tn+fp)   if (tn+fp) >0 else 0
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
    

    # 打印结果
    print("Accuracy:", round(accuracy,4))
    print("AUC",round(auc,4))
    print("MCC",round(mcc,4))
    print("Precision:", round(precision,4))
    print("Recall:", round(recall,4))
    print("F1 Score:", round(f1_score,4))
    print("Positive Accuracy:",round(p_acc,4))
    print("Negative Accuracy:",round(n_acc,4))
    return accuracy,auc


def calculate_binary_classification_metrics(tp, fp, tn, fn,prob,label):
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    auc = roc_auc_score(label, prob)
    p_acc=tp/(tp+fn) if (tp+fn) >0 else 0
    n_acc=tn/(tn+fp)   if (tn+fp) >0 else 0
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
    

    results = {
                "Accuracy": round(accuracy, 4),
                "AUC": round(auc, 4),
                "MCC": round(mcc, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1 Score": round(f1_score, 4),
                "Positive Accuracy": round(p_acc, 4),
                "Negative Accuracy": round(n_acc, 4),
                }
    return results


def check_binary_classification_metrics(prob_list,label_list,thread):
    tp=0
    fp=0
    tn=0
    fn=0
    pre_list=[]
    for prob in prob_list:
        if prob>=thread:
            pre_list.append(1)
        else:
            pre_list.append(0)
    for i in range(len(pre_list)):
        pre=pre_list[i]
        label=label_list[i]
        if pre==1 and label==1:tp+=1
        if pre==1 and label==0:fp+=1
        if pre==0 and label==0:tn+=1
        if pre==0 and label==1:fn+=1
        
        
        
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    auc = roc_auc_score(label_list, prob_list)
    p_acc=tp/(tp+fn) if (tp+fn) >0 else 0
    n_acc=tn/(tn+fp)   if (tn+fp) >0 else 0
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
    

    # 打印结果
    print("Accuracy:", round(accuracy,4))
    print("AUC",round(auc,4))
    print("MCC",round(mcc,4))
    print("Precision:", round(precision,4))
    print("Recall:", round(recall,4))
    print("F1 Score:", round(f1_score,4))
    print("Positive Accuracy:",round(p_acc,4))
    print("Negative Accuracy:",round(n_acc,4))


   
    return accuracy,auc


import torch

def threshold_predict(tensor, threshold):

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if tensor.ndim != 1:
        raise ValueError("Input tensor must be one-dimensional")
    
    # 根据阈值转换
    binary_tensor = (tensor > threshold).int()  # 转为 0 或 1 的 Tensor
    return binary_tensor.tolist()  # 转为 Python 列表


def calculate_metrics(outputs, labels):
    outputs = outputs.long()
    labels = labels.long()

    # 计算TP, FP, FN, TN
    tp = torch.sum((outputs == 1) & (labels == 1)).item()
    fp = torch.sum((outputs == 1) & (labels == 0)).item()
    fn = torch.sum((outputs == 0) & (labels == 1)).item()
    tn = torch.sum((outputs == 0) & (labels == 0)).item()

    return tp, fp, fn, tn