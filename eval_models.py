#!/usr/bin/env python3
"""快速评估Teacher和Student模型的精度"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_model(model, dataloader, device='cuda'):
    """评估模型在数据集上的精度"""
    model.eval()
    model.to(device)
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            emg = batch["emg"].to(device)
            targets = batch["targets"].to(device)
            
            # 提取真实标签
            y = targets.max(dim=2)[0].argmax(dim=1)
            
            # 模型预测
            if hasattr(model, 'forward'):
                # Student模型直接输出logits
                if emg.shape != torch.Size([emg.shape[0], 16, 16000]):
                    # 可能需要转置
                    pass
                try:
                    logits = model(emg)
                    if logits.dim() == 3:
                        # Teacher输出是[batch, classes, time]
                        logits = logits.mean(dim=2)
                except:
                    # Teacher模型可能需要不同处理
                    logits = model(emg)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    if logits.dim() == 3:
                        logits = logits.mean(dim=2)
            
            # 计算准确率
            predictions = logits.argmax(dim=1)
            correct = (predictions == y).sum().item()
            
            total_correct += correct
            total_samples += y.size(0)
    
    accuracy = total_correct / total_samples
    return accuracy

if __name__ == "__main__":
    print("请在Jupyter Notebook中使用此评估脚本")
    print("\n示例用法:")
    print("  from eval_models import evaluate_model")
    print("  teacher_acc = evaluate_model(teacher, data_module.val_dataloader())")
    print("  student_acc = evaluate_model(student, data_module.val_dataloader())")
