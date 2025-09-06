import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from Dataset import DisasterTweetsDataset
from config import TRAIN_CSV_PATH, TEST_CSV_PATH, TRAIN_SPLIT_RATIO, BATCH_SIZE, NUM_WORKERS, SHUFFLE, RANDOM_SEED


def get_train_val_dataloaders(validation_split=0.2, shuffle=True, random_seed=None):
    """
    获取训练和验证数据加载器
    
    :param validation_split: 验证集比例
    :param shuffle: 是否打乱数据
    :param random_seed: 随机种子
    :return: train_loader, val_loader
    """
    # 创建完整训练数据集
    dataset = DisasterTweetsDataset(TRAIN_CSV_PATH)
    
    # 获取数据集大小
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # 计算分割点
    split = int(np.floor(validation_split * dataset_size))
    
    # 设置随机种子
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 打乱索引
    if shuffle:
        np.random.shuffle(indices)
    
    # 分割索引
    train_indices, val_indices = indices[split:], indices[:split]
    
    # 创建采样器
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, 
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler, 
                            num_workers=NUM_WORKERS)
    
    return train_loader, val_loader


def get_test_dataloader():
    """
    获取测试数据加载器
    
    :return: test_loader
    """
    # 创建测试数据集
    dataset = DisasterTweetsDataset(TEST_CSV_PATH)
    
    # 创建数据加载器
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=NUM_WORKERS)
    
    return test_loader


if __name__ == "__main__":
    # 示例用法
    train_loader, val_loader = get_train_val_dataloaders(
        validation_split=1-TRAIN_SPLIT_RATIO, 
        shuffle=SHUFFLE, 
        random_seed=RANDOM_SEED
    )
    
    test_loader = get_test_dataloader()
    
    print(f"训练批次数量: {len(train_loader)}")
    print(f"验证批次数量: {len(val_loader)}")
    print(f"测试批次数量: {len(test_loader)}")
    
    # 显示一个训练批次的样本
    for batch in train_loader:
        print("训练批次样本:")
        print(f"  文本: {batch['text'][0][:50]}...")  # 显示前50个字符
        print(f"  特征形状: {batch['features'].shape}")
        print(f"  标签: {batch['label'][:5]}")  # 显示前5个标签
        break