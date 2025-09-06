import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import numpy as np
import logging
import os
import time
from datetime import datetime

from model.TextModel import TextModel
from model.FeatureModel import FeatureModel
from model.Model import NLPGettingStartedNet
from Dataloader import get_train_val_dataloaders, get_test_dataloader
from Dataset import DisasterTweetsDataset
from config import TRAIN_CSV_PATH, TEST_CSV_PATH, BATCH_SIZE, RANDOM_SEED

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def compute_metrics(predictions, labels):
    """计算评估指标"""
    # 将概率转换为二进制预测
    binary_predictions = (predictions > 0.5).astype(int)
    
    # 计算各种指标
    f1 = f1_score(labels, binary_predictions)
    accuracy = accuracy_score(labels, binary_predictions)
    auc = roc_auc_score(labels, predictions)
    
    return f1, accuracy, auc


def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=20, save_path='best_model.pt'):
    """训练模型"""
    logger.info(f"Starting model training for {num_epochs} epochs")
    
    # 初始化最佳验证AUC和F1-score
    best_val_auc = 0.0
    best_val_f1 = 0.0
    
    # 训练循环
    for epoch in range(num_epochs):
        logger.info(f"===== Epoch {epoch+1}/{num_epochs} =====")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []
        
        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            # 获取数据
            input_ids = batch['text']['input_ids'].to(device)
            attention_mask = batch['text']['attention_mask'].to(device)
            features = batch['features'].float().to(device)
            labels = batch['label'].float().to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(input_ids, attention_mask, features).squeeze()
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累积损失和预测
        train_loss += loss.item()
        # 应用sigmoid激活函数获取概率值用于评估
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        train_predictions.extend(probs)
        train_labels.extend(labels.detach().cpu().numpy())
        
        # 计算训练指标
        train_time = time.time() - start_time
        train_f1, train_accuracy, train_auc = compute_metrics(np.array(train_predictions), np.array(train_labels))
        avg_train_loss = train_loss / len(train_loader)
        
        logger.info(f"Training Results - Loss: {avg_train_loss:.4f} | AUC: {train_auc:.4f} | Accuracy: {train_accuracy:.4f} | F1-Score: {train_f1:.4f} | Time: {train_time:.2f}s")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # 获取数据
                input_ids = batch['text']['input_ids'].to(device)
                attention_mask = batch['text']['attention_mask'].to(device)
                features = batch['features'].float().to(device)
                labels = batch['label'].float().to(device)
                
                # 前向传播
                outputs = model(input_ids, attention_mask, features).squeeze()
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 累积损失和预测
                val_loss += loss.item()
                # 应用sigmoid激活函数获取概率值用于评估
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                val_predictions.extend(probs)
                val_labels.extend(labels.detach().cpu().numpy())
        
        # 计算验证指标
        val_f1, val_accuracy, val_auc = compute_metrics(np.array(val_predictions), np.array(val_labels))
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Validation Results - Loss: {avg_val_loss:.4f} | AUC: {val_auc:.4f} | Accuracy: {val_accuracy:.4f} | F1-Score: {val_f1:.4f}")
        
        # 保存最佳模型（基于AUC或F1-score）
        if False and val_auc > best_val_auc:
            best_val_auc = val_auc
            # 保存模型
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model (AUC: {val_auc:.4f}, F1: {val_f1:.4f})")
        
        # 也可以选择基于F1-score保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model (AUC: {val_auc:.4f}, F1: {best_val_f1:.4f})")
    
    logger.info("Training completed")
    return model


# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)


# 定义collate_fn来处理文本数据
def collate_fn(batch):
    # 处理文本数据
    texts = [item['text'] for item in batch]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    
    # 处理特征和标签
    features = torch.stack([torch.tensor(item['features'], dtype=torch.float) for item in batch])
    
    if 'label' in batch[0]:
        labels = torch.stack([torch.tensor(item['label'], dtype=torch.float) for item in batch])
        return {
            'text': encodings,
            'features': features,
            'label': labels
        }
    else:
        return {
            'text': encodings,
            'features': features
        }


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建数据集和数据加载器
    logger.info("加载数据集...")
    train_loader, val_loader = get_train_val_dataloaders()
    
    # 获取特征维度
    dataset = DisasterTweetsDataset(TRAIN_CSV_PATH)
    feature_dim = len(dataset.get_feature_names())
    logger.info(f"特征维度: {feature_dim}")
    
    # 更新数据加载器的collate_fn
    train_loader = DataLoader(train_loader.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_loader.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    # 初始化模型
    logger.info("初始化模型...")
    text_model = TextModel()
    feature_model = FeatureModel(feature_dim=feature_dim)
    model = NLPGettingStartedNet(text_model, feature_model)
    
    # 将模型移到设备上
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失
    # 只优化未冻结的参数，使用更小的学习率
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    
    # 训练模型
    model = train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=20, save_path='best_model.pt')
    
    logger.info("模型训练完成，最佳模型已保存")


if __name__ == "__main__":
    main()