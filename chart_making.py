import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from utils import Config

config = Config("./config/config.yml")
result_path = config["result_path"]
assert os.path.exists(result_path), f"{result_path} does not exist"
all_models = os.listdir(result_path)
all_models = [model for model in all_models if
              os.path.exists(os.path.join(result_path, model, 'test_0'))]
all_model_path = [os.path.join(os.path.join(result_path, model)) for model in all_models]

def training_loss_chart():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
    colors = plt.cm.tab10((np.linspace(0, 1, len(all_models))))
    for idx, (model_path, model_name) in enumerate(zip(all_model_path, all_models)):
        train_loss_path = os.path.join(model_path, "test_0", "train_loss.txt")
        val_loss_path = os.path.join(model_path, "test_0", "val_loss.txt")
        with open(train_loss_path, 'r') as f:
            train_losses = eval(f.read())
        with open(val_loss_path, 'r') as f:
            val_losses = eval(f.read())

        epoches = range(1, len(train_losses) + 1)
        ax1.plot(epoches, train_losses, color=colors[idx], label=model_name)
        ax2.plot(epoches, val_losses, color=colors[idx], label=model_name)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    total_result_path = os.path.join(result_path, 'total/')
    if not os.path.exists(total_result_path):
        os.makedirs(total_result_path)
    save_path = os.path.join(result_path, 'total/', 'training_loss_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"figure saved to: {save_path}")

    plt.show()

def result_loss_chart():
    loss_table = pd.DataFrame(columns=['test_loss', 'best_val_loss', 'best_train_loss'])
    for idx, (model_path, model_name) in enumerate(zip(all_model_path, all_models)):
        test_result_path = os.path.join(model_path, "test_results.csv")
        df = pd.read_csv(test_result_path)
        ave_test_loss = df["test_loss"].mean()
        ave_best_val_loss = df["best_val_loss"].mean()
        ave_best_train_loss = df["best_train_loss"].mean()
        df = pd.DataFrame([[ave_test_loss, ave_best_val_loss, ave_best_train_loss]],
                          columns=['test_loss', 'best_val_loss', 'best_train_loss'],
                          index=[model_name])
        loss_table = pd.concat([loss_table, df])
    print(loss_table)

    fig, ax = plt.subplots()
    colors = plt.cm.tab10((np.linspace(0, 1, len(all_models))))
    loss_table.T.plot(ax=ax, kind='bar', width=0.8, color=colors)
    ax.set_xlabel('criterion', fontsize=12)
    plt.xticks(rotation=0, fontsize=11)
    ax.set_ylabel('loss', fontsize=12)
    ax.set_title('result_comparison', fontsize=14, fontweight='bold')
    ax.legend(title='model', bbox_to_anchor=(1.05, 1))
    ax.grid(True, alpha=0.3, axis='y')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=8)
    total_result_path = os.path.join(result_path, 'total/')
    if not os.path.exists(total_result_path):
        os.makedirs(total_result_path)
    save_path = os.path.join(result_path, 'total/', 'result_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

training_loss_chart()
result_loss_chart()

