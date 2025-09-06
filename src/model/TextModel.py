import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextModel(nn.Module):
    """使用DeBERTa-v3-base模型处理文本数据的模型"""
    
    def __init__(self, model_name="microsoft/deberta-v3-base", num_classes=64):
        """
        初始化模型
        :param model_name: 预训练模型名称
        :param num_classes: 分类数量
        """
        
        super(TextModel, self).__init__()

        self.output_dim = num_classes

        
        # 加载预训练的DeBERTa-v3-base模型
        self.deberta = AutoModel.from_pretrained(model_name)
        
        # 分类头
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_classes)
        
        # Sigmoid激活函数用于二分类
        
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        :param input_ids: 输入token ID
        :param attention_mask: 注意力掩码
        :return: 模型输出
        """
        # 获取DeBERTa模型的输出
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS] token的输出进行分类
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # [CLS] token的表示
        
        # 分类
        logits = self.classifier(cls_output)
        
        # 应用Sigmoid激活函数
        
        return logits