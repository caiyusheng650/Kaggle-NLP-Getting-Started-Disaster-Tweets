import torch
import torch.nn as nn


class FeatureModel(nn.Module):
    """处理文本以外特征数据的模型"""
    
    def __init__(self, feature_dim, embedding_dim=64, hidden_dim=128, output_dim=64):
        """
        初始化模型
        :param feature_dim: 输入特征维度
        :param embedding_dim: embedding维度
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出维度
        """
        super(FeatureModel, self).__init__()
        self.output_dim = output_dim

        # 特征embedding层
        self.feature_embedding = nn.Linear(feature_dim, embedding_dim)

        
        # 隐藏层
        self.hidden_layer = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, features):
        """
        前向传播
        :param features: 输入特征 (batch_size, feature_dim)
        :return: 输出结果 (batch_size, output_dim)
        """
        # 特征embedding
        embedded_features = self.feature_embedding(features)
        
        # 隐藏层
        output = self.relu(self.hidden_layer(embedded_features))

        output = self.output_layer(output)

        
        # 输出层        
        return output