import torch
import torch.nn as nn


class NLPGettingStartedNet(nn.Module):
    """结合文本特征模型和非文本特征模型的综合模型"""
    
    def __init__(self, text_model, feature_model, combined_hidden_dim=128, num_classes=1):
        """
        初始化模型
        :param text_model: 文本模型实例
        :param feature_model: 特征模型实例
        :param combined_hidden_dim: 组合层隐藏维度
        :param num_classes: 分类数量
        """
        super(NLPGettingStartedNet, self).__init__()
        
        # 文本模型和特征模型
        self.text_model = text_model
        self.feature_model = feature_model
        
        # 获取特征模型的输出维度
        feature_model_output_dim = feature_model.output_dim

        text_model_output_dim = text_model.output_dim

        
        # 组合层
        # 文本模型输出维度为1，特征模型输出维度为feature_output_dim
        self.combined_layer = nn.Linear(text_model_output_dim + feature_model_output_dim, combined_hidden_dim)
        self.relu = nn.ReLU()
        
        # 分类头
        self.classifier = nn.Linear(combined_hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask, features):
        """
        前向传播
        :param input_ids: 文本token ID
        :param attention_mask: 文本注意力掩码
        :param features: 非文本特征
        :return: 模型输出
        """
        # 获取文本模型和特征模型的输出
        text_output = self.text_model(input_ids, attention_mask)
        feature_output = self.feature_model(features)

        # 组合两个模型的输出
        combined_input = torch.cat((text_output, feature_output), dim=1)

        
        # 组合层
        combined_output = self.relu(self.combined_layer(combined_input))
        
        # 分类
        logits = self.classifier(combined_output)
        
        
        return logits