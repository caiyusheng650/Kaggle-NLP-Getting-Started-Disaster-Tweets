# 数据集配置
TRAIN_CSV_PATH = 'data/train_cleaned.csv'
TEST_CSV_PATH = 'data/test_cleaned.csv'

# 训练配置
TRAIN_SPLIT_RATIO = 0.8  # 训练集占比
BATCH_SIZE = 32
NUM_WORKERS = 2
SHUFFLE = True

# 随机种子
RANDOM_SEED = 42