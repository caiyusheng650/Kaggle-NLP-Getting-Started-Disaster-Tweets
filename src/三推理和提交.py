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
import pandas as pd
from config import TEST_CSV_PATH, BATCH_SIZE, RANDOM_SEED

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/submission.log'),
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


def test_model(model, test_loader, device):
    """测试模型"""
    logger.info(f"Starting model testing")
    
    # 验证阶段
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 获取数据
            input_ids = batch['text']['input_ids'].to(device)
            attention_mask = batch['text']['attention_mask'].to(device)
            features = batch['features'].float().to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask, features).squeeze()
                        
            probs = (torch.sigmoid(outputs).detach().cpu().numpy()>0.5).astype(int)
            predictions.extend(probs)
    logger.info(f"Testing completed. Predicted samples: {len(predictions)}")
    return predictions


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
    logger.info(f"Using device: {device}")
    
    # 创建数据集和数据加载器
    logger.info("Loading dataset...")
    test_loader = get_test_dataloader()
    
    # 获取特征维度
    dataset = DisasterTweetsDataset(TEST_CSV_PATH)
    feature_dim = len(dataset.get_feature_names())
    
    # 更新数据加载器的collate_fn
    test_loader = DataLoader(test_loader.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    # 初始化模型
    logger.info("Loading model...")

    text_model = TextModel()
    feature_model = FeatureModel(feature_dim=feature_dim)
    model = NLPGettingStartedNet(text_model, feature_model)
    
    # 将模型移到设备上
    model.to(device)
    model.load_state_dict(torch.load("best_model.pt"))

    
    # 训练模型
    predictions = test_model(model, test_loader, device)

    df_submission = pd.read_csv("data/nlp-getting-started/sample_submission.csv")
    df_submission['target'] = predictions
    df_submission[['id', 'target']].to_csv('submission.csv', index=False)

    # 本项目在Kaggle排行榜上当前取得F1score 0.82224


if __name__ == "__main__":
    main()