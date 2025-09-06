import pandas as pd
import torch
from torch.utils.data import Dataset


class DisasterTweetsDataset(Dataset):
    """灾难推文数据集类，用于加载和处理训练/测试数据"""
    
    def __init__(self, csv_file, transform=None):
        """
        初始化数据集
        :param csv_file: CSV文件路径
        :param transform: 可选的转换函数
        """        
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # 确定是否为测试集（没有target列）
        self.is_test = 'target' not in self.data.columns
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取指定索引的数据项"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 获取文本数据
        text = self.data.iloc[idx]['text_cleaned']
        
        # 获取特征数据（除text_cleaned和target外的所有列）
        feature_columns = [col for col in self.data.columns if col not in ['text_cleaned', 'target']]
        features = self.data.iloc[idx][feature_columns].values.astype(float)
        
        # 如果是测试集，只返回文本和特征
        if self.is_test:
            sample = {
                'text': text,
                'features': features
            }
        else:
            # 获取标签
            label = self.data.iloc[idx]['target']
            sample = {
                'text': text,
                'features': features,
                'label': label
            }
        
        # 应用转换
        if self.transform:
            sample = self.transform(sample)
            
        return sample

    def get_feature_names(self):
        """获取特征列名称"""
        return [col for col in self.data.columns if col not in ['text_cleaned', 'target']]