import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import time
from models.Transformer import Transformer
from utils import Config
from utils import WeatherDataset
import seaborn as sns
import matplotlib.pyplot as plt

config = Config("./config/config.yml")

# ===================== 数据读取与预处理=====================
# 构建数据集和数据加载器
train_dataset = WeatherDataset(config, mode="train")
val_dataset = WeatherDataset(config, mode="val")
test_dataset = WeatherDataset(config, mode="test")

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
# ===================== 模型训练与评估（匹配文档损失函数和优化器）=====================
def train_model(config):
    # 初始化模型、损失函数、优化器
    if config['model'] == 'Transformer':
        model = Transformer(config)
    else:
        print('invalid model')
        return None, None, None
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    result_path = os.path.join(config["result_path"], config["model"])
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    test_result_path = os.path.join(result_path, "test_results.csv")
    if not os.path.exists(test_result_path):
        full_test_results = pd.DataFrame(columns=(list(config.get_all_keys())+
                                                  ['param_sum','ave_runtime', 'best_val_loss','test_loss']))
        test_id = 0
    else:
        full_test_results = pd.read_csv(test_result_path)
        test_id = full_test_results.shape[0]
    result_path = os.path.join(result_path, f'test_{test_id}')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    best_val_loss = float('inf')
    best_train_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []
    print(f"begin training on {config["device"]}...")
    for epoch in range(config["epochs"]):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_data, batch_label in train_loader:
            batch_data = batch_data.to(config["device"])
            batch_label = batch_label.to(config["device"])

            optimizer.zero_grad()
            pred = model(batch_data)
            loss = criterion(pred, batch_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()

            train_loss += loss.item() * batch_data.shape[0]

        train_loss /= len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_label in val_loader:
                batch_data = batch_data.to(config["device"])
                batch_label = batch_label.to(config["device"])
                pred = model(batch_data)
                loss = criterion(pred, batch_label)
                val_loss += loss.item() * batch_data.shape[0]

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{config["epochs"]}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.8f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(result_path, "best_model.pth"))
        if train_loss < best_train_loss:
            best_train_loss = train_loss

    plt.title = 'train loss'
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_losses, color='b', label='train')
    plt.plot(val_losses, color='r', label='val')
    plt.legend()
    plt.savefig(os.path.join(result_path, "train_loss.png"))
    plt.close()
    #plt.show()
    with open(os.path.join(result_path, "train_loss.txt"),'w') as f:
        f.write(str(train_losses))
    with open(os.path.join(result_path, "val_loss.txt"),'w') as f:
        f.write(str(val_losses))


    # 测试阶段
    model.load_state_dict(torch.load(os.path.join(result_path, "best_model.pth")))
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_labels = []
    times = 0
    with torch.no_grad():
        for batch_data, batch_label in test_loader:
            start_time = time.time()
            batch_data = batch_data.to(config["device"])
            batch_label = batch_label.to(config["device"])
            pred = model(batch_data)
            loss = criterion(pred, batch_label)
            test_loss += loss.item() * batch_data.shape[0]

            # 反归一化

            pred = pred.cpu() * (train_dataset.std + 1e-8) + train_dataset.mean
            batch_label = batch_label.cpu() * (train_dataset.std + 1e-8) + train_dataset.mean

            test_preds.append(pred.numpy())
            test_labels.append(batch_label.cpu().numpy())
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            times += (end_time - start_time) * 1000 * batch_data.shape[0]


    test_loss /= len(test_loader.dataset)
    test_preds = np.concatenate(test_preds, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    print(f"test loss：{test_loss:.6f}")
    total_params = sum(p.numel() for p in model.parameters())
    ave_runtime = times / len(test_loader.dataset)
    informations = config.get_all_values() + [total_params, ave_runtime, best_train_loss, best_val_loss, test_loss]
    informations = pd.DataFrame([informations], columns=(list(config.get_all_keys())+
                                                  ['param_sum','ave_runtime','best_train_loss','best_val_loss','test_loss']))
    full_test_results = pd.concat([full_test_results, informations],ignore_index=True)
    print(full_test_results)
    full_test_results.to_csv(test_result_path, index=False)
    np.save(os.path.join(result_path, "test_predicts.pth"), test_preds)
    np.save(os.path.join(result_path, "test_lables.pth"), test_labels)
    return best_val_loss, test_loss




def predict_future(model, config, input_seq):
    """
    输入：历史12步数据（1, 12, 13, 15, 12）
    返回：未来12步预测结果（1, 12, 13, 15, 12）
    """
    model.eval()
    input_seq = torch.tensor(input_seq, dtype=torch.float32).to(config["device"])
    with torch.no_grad():
        pred = model(input_seq)
    # 反归一化
    pred = pred.cpu().numpy() * (train_dataset.std + 1e-8) + train_dataset.mean
    return pred


# ===================== 主函数执行=====================
if __name__ == "__main__":
    val_losses = []
    test_losses = []
    for i in range(config['experiment_rounds']):
        val_loss, test_loss = train_model(config)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
    for i in range(config['experiment_rounds']):
        print(f'experiment_round: {i} : val_loss: {val_losses[i]:.6f}, test_loss: {test_losses[i]:6f}')
    print(f'average : val_loss:{np.mean(val_losses)}, test_loss: {np.mean(test_losses)}')