# NLP入门实战——灾难推文分类

## 项目结构

```
├── data/               # 数据集目录
│   ├── nlp-getting-started/  # 原始比赛数据
│   ├── *_cleaned.csv   # 清洗后的数据集
├── src/                # 源代码目录
│   ├── Dataloader.py   # 数据加载器实现
│   ├── Dataset.py      # 自定义数据集类
│   ├── config.py       # 训练配置参数
│   ├── model/          # 模型架构
│   │   ├── TextModel.py    # 文本编码主干网络
│   │   ├── FeatureModel.py # 特征工程模块
│   │   └── Model.py       # 完整分类模型
│   ├── 一洞察性数据分析.ipynb  # 数据探索分析
│   ├── 二模型构建和训练.py    # 模型训练入口
│   └── 三推理和提交.py      # 生成预测结果
├── best_model.pt       # 训练好的最佳模型
├── requirements.txt    # Python依赖库
└── setup.py            # 项目安装配置
```

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 准备数据
- 将原始数据解压到data/nlp-getting-started目录
- 运行数据分析笔记本来理解数据分布

3. 训练模型
```bash
python src/二模型构建和训练.py \
    --data_dir data/train_cleaned.csv \
    --model_save best_model.pt
```

4. 生成提交
```bash
python src/三推理和提交.py \
    --test_data data/test_cleaned.csv \
    --model_path best_model.pt \
    --output submission.csv
```

## 代码模块说明

### 核心组件
- Dataloader：实现批处理数据加载，支持文本编码和特征工程流水线
- Dataset：自定义数据集类，整合文本清洗和特征提取
- Model：集成文本编码器和分类器的端到端模型

### 训练流程
- 一洞察性数据分析：包含可视化分析和数据预处理方案
- 二模型构建和训练：包含训练循环、验证指标计算和模型保存功能
- 三推理和提交：加载训练好的模型生成最终预测结果
- config：集中管理超参数（学习率、批大小等）

## 数据集说明
数据来源：Kaggle灾难推文分类竞赛

## 评估指标
使用F1-score作为评估标准：
```
F1 = 2 * (precision * recall) / (precision + recall)
```
