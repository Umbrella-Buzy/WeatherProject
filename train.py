import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sympy.physics.units import momentum
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from tqdm import tqdm
import time
from models.Transformer import Transformer
from utils.config import Config
from utils.data_process import *
from utils.training_utils import *
import matplotlib.pyplot as plt

os.environ["USE_LIBUV"] = "0"
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '8800'
os.environ["USE_LIBUV"] = "0"
config = Config("./config/config.yml")

# ===================== 数据读取与预处理=====================
# 构建数据集和数据加载器
trainer = DistributedTrainer(config, )
device = trainer.get_device()

train_dataset = WeatherDataset(config, mode="train")
test_dataset = WeatherDataset(config, mode="test")
if trainer.is_main_process():
    print(f'loading training data...')
train_loader = trainer.get_dataloader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
if trainer.is_main_process():
    print(f'loading test data...')
test_loader = trainer.get_dataloader(test_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(trainer.model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
full_test_results, test_id, result_path, test_result_path = init_results(config)

# ===================== 模型训练与评估（匹配文档损失函数和优化器）=====================
def train_and_test_model(config):
    train_losses = []
    best_train_loss = float('inf')
    if trainer.is_main_process():
        print(f"begin training on {device}...")
    for epoch in range(config["epochs"]):
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        trainer.model.train()
        train_loss = 0.0
        m_train_loader = train_loader
        if trainer.is_main_process():
            m_train_loader = tqdm(m_train_loader)
        for _, (batch_data, batch_dec_input, batch_label) in enumerate(m_train_loader):
            batch_data = batch_data.to(device)
            batch_dec_input = batch_dec_input.to(device)
            batch_label = batch_label.to(device)

            optimizer.zero_grad()
            pred = trainer.model(batch_data, batch_dec_input)
            loss = criterion(pred, batch_label)
            loss.backward()
            #loss = trainer.reduce_loss(loss)

            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), config["max_grad_norm"])
            optimizer.step()

            #train_loss += loss.item() * batch_data.shape[0]

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        if best_train_loss > train_loss:
            best_train_loss = train_loss
        if trainer.is_main_process():
            print(f"epoch {epoch} train loss: {train_loss}")
        if epoch % config["save_epochs"] == 0:
            trainer.save_model(os.path.join(result_path, f'best_model.pth'))
            torch.cuda.empty_cache()
    trainer.save_model(os.path.join(result_path, f'best_model.pth'))
    torch.cuda.empty_cache()

    plt.title = 'train loss'
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_losses, color='b', label='train')
    plt.legend()
    plt.savefig(os.path.join(result_path, "train_loss.png"))
    plt.close()
    #plt.show
    if trainer.is_main_process():
        with open(os.path.join(result_path, "train_loss.txt"),'w') as f:
            f.write(str(train_losses))

    # 测试阶段
    trainer.load_model(os.path.join(result_path, "best_model.pth"))
    trainer.model.eval()
    test_loss = 0
    times = 0
    correct_counts = [0] * config['pred_steps']
    for _, (batch_data, _, batch_label) in enumerate(test_loader):
        start_time = time.time()
        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)
        pred = trainer.model.generate(batch_data)
        loss = criterion(pred, batch_label)
        loss = trainer.reduce_loss(loss)
        correct_counts = [s+c for s,c in zip(correct_counts, get_correct(pred, batch_label))]
        test_loss += loss.item() * batch_data.shape[0]
        end_time = time.time()
        times += (end_time - start_time) * 1000 * batch_data.shape[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    total_len = len(test_loader.dataset)
    test_loss /= total_len
    accuracy = [x / total_len for x in correct_counts]
    accuracy = trainer.reduce_loss(accuracy)
    total_accuracy = sum(accuracy) / config['pred_steps']
    if trainer.is_main_process():
        print(f"test loss：{test_loss:.6f}")
        print(accuracy)
        print(f"total accuracy {total_accuracy}")
    total_params = sum(p.numel() for p in trainer.model.parameters())
    ave_runtime = times / len(test_loader.dataset)
    informations = config.get_all_values() + [total_params, ave_runtime, best_train_loss, test_loss, total_accuracy]
    informations = pd.DataFrame([informations], columns=(list(config.get_all_keys())+
                                                  ['param_sum','ave_runtime','best_train_loss','test_loss','test_accuracy']))
    n_full_test_results = pd.concat([full_test_results, informations],ignore_index=True)
    if trainer.is_main_process():
        print(n_full_test_results)
    n_full_test_results.to_csv(test_result_path, index=False)
    if trainer.is_main_process():
        with open(os.path.join(result_path, "accuracy.txt"),'w') as f:
            f.write(str(accuracy))
    trainer.cleanup()
    return test_loss, total_accuracy

def test():
    trainer.load_model(os.path.join("./results/Transformer/test_4/", "best_model.pth"))
    trainer.model.eval()
    test_loss = 0
    times = 0
    correct_counts = [0] * config['pred_steps']
    for batch_data, batch_dec_input, batch_label in tqdm(test_loader):
        start_time = time.time()
        batch_data = batch_data.to(device)
        batch_dec_input = batch_dec_input.to(device)
        batch_label = batch_label.to(device)
        pred = trainer.model.generate(batch_data)
        #pred = trainer.model(batch_data, batch_dec_input)
        loss = criterion(pred, batch_label)
        loss = trainer.reduce_loss(loss)
        correct_counts = [s + c for s, c in zip(correct_counts, get_correct(pred, batch_label))]
        test_loss += loss.item() * batch_data.shape[0]
        end_time = time.time()
        times += (end_time - start_time) * 1000 * batch_data.shape[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    total_len = len(test_loader.dataset)
    test_loss /= total_len
    accuracy = [x / total_len for x in correct_counts]
    accuracy = trainer.reduce_loss(accuracy)
    total_accuracy = sum(accuracy) / config['pred_steps']
    if trainer.is_main_process():
        print(f"test loss：{test_loss:.6f}")
        print(accuracy)
        print(f"total accuracy {total_accuracy}")


# ===================== 主函数执行=====================
if __name__ == "__main__":
    test_losses = []
    test_accuracies = []
    for i in range(config['experiment_rounds']):
        test_loss, test_accuracy = train_and_test_model(config)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    for i in range(config['experiment_rounds']):
        print(f'experiment_round: {i} : test_loss: {test_losses[i]:6f}, test_accuracy: {test_accuracies[i]:6f}')
    print(f'average : test_loss: {np.mean(test_losses)}, test_accuracy: {np.mean(test_accuracies)}')
    '''
    test()
    '''