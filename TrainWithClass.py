import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from sympy.physics.units import momentum
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from tqdm import tqdm
import time
from models.Transformer import Transformer, DirectTransformer
from models.preprocessor import preprocessor
from utils.config import Config
from utils.data_process import *
from utils.training_utils import *
import matplotlib.pyplot as plt

config = Config("./config/config.yml")

class ModelTrainer:
    def __init__(self, config, intro=None):
        self._save = True
        self._print_message = True

        self.config = config
        self.intro = intro
        self.device = self.config["device"]
        self.model = self._prepare_model()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"],
                               weight_decay=config["weight_decay"])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, threshold=1e-4,
                                            cooldown=0,  min_lr=self.config["min_lr"])
        self.preprocessor = preprocessor(config).to(self.device)

        self.result_path = init_results(config)
        self.test_params = pd.DataFrame(columns=(list(config.get_all_keys())+
                                        ['param_sum','ave_runtime','best_train_loss','test_loss','test_accuracy']))

        self.best_train_loss = float('inf')

        if self.intro is not None:
            with open(os.path.join(self.result_path, "intro.txt"), "w") as f:
                f.write(intro)


        if self._print_message:
            print(f'loading training data...')
        train_dataset = WeatherDataset(config, mode="train")
        self.train_loader = self.get_dataloader(train_dataset, shuffle=True, drop_last=True)
        self._i_train_loader = self.train_loader

        if self._print_message:
            print(f'loading testing data...')
        test_dataset = WeatherDataset(config, mode="test")
        self.test_loader = self.get_dataloader(test_dataset, shuffle=False, drop_last=False)
        #self._i_test_loader = tqdm(self.test_loader) if self._print_message else self.test_loader

    def _prepare_model(self):
        if self.config['model'] == 'Transformer':
            model = Transformer(config).to(self.device)
        if self.config['model'] == 'DirectTransformer':
            model = DirectTransformer(config).to(self.device)
        else:
            print('invalid model')
            self._has_error = True
            return None
        return model

    def get_dataloader(self, dataset, shuffle=True, drop_last=False):
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
            pin_memory=True,
            num_workers=0,
            drop_last=drop_last
            )
        return dataloader

    def save_model(self, path):
        if self._save:
            state_dict = self.model.state_dict()
            torch.save(state_dict, path)

    def load_model(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    def train(self):
        train_losses = []
        if self._print_message:
            print(f"begin training on {self.device}...")
        for epoch in range(config["epochs"]):
            self.model.train()
            train_loss = 0.0
            #self._i_train_loader = tqdm(self.train_loader) if self._print_message else self.train_loader
            for batch_data, batch_dec_input, batch_label in tqdm(self.train_loader):
                batch_data = self.preprocessor(batch_data.to(self.device))
                batch_dec_input = self.preprocessor(batch_dec_input.to(self.device))
                batch_label = self.preprocessor(batch_label.to(self.device))

                self.optimizer.zero_grad()
                pred = self.model(batch_data, batch_dec_input)
                loss = self.criterion(pred, batch_label)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                self.optimizer.step()
                train_loss += loss.item() * batch_data.shape[0]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            train_loss /= len(self.train_loader.dataset)
            train_losses.append(train_loss)
            if self.best_train_loss > train_loss:
                self.best_train_loss = train_loss
            if self._print_message:
                print(f"epoch {epoch} train loss: {train_loss}")
            if epoch % config["save_epochs"] == 0 and self._save:
                self.save_model(os.path.join(self.result_path, f'best_model.pth'))
                torch.cuda.empty_cache()
        if self._save:
            self.save_model(os.path.join(self.result_path, f'best_model.pth'))
        torch.cuda.empty_cache()

        plt.title = 'train loss'
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(train_losses, color='b', label='train')
        plt.legend()
        plt.savefig(os.path.join(self.result_path, "train_loss.png"))
        plt.close()
        # plt.show
        if self._save:
            with open(os.path.join(self.result_path, "train_loss.txt"), 'w') as f:
                f.write(str(train_losses))

    def test(self, path=None):
        if path is not None:
            self.load_model(os.path.join(path, "best_model.pth"))
        else:
            self.load_model(os.path.join(self.result_path, "best_model.pth"))
        self.model.eval()
        test_loss = 0
        times = 0
        correct_counts = np.zeros((3, self.config["pred_steps"]))
        record_label = None
        record_prediction = None
        for batch_data, batch_dec_input, batch_label in tqdm(self.test_loader):
            batch_data = self.preprocessor(batch_data.to(self.device))
            batch_dec_input = self.preprocessor(batch_dec_input.to(self.device))
            batch_label = self.preprocessor(batch_label.to(self.device))
            start_time = time.time()
            pred = self.model.generate(batch_data)
            # pred = trainer.model(batch_data, batch_dec_input)
            record_label = batch_label[0]
            record_prediction = pred[0]
            loss = self.criterion(pred, batch_label)
            correct_counts += get_correct(pred, batch_label)
            test_loss += loss.item() * batch_data.shape[0]
            end_time = time.time()
            times += (end_time - start_time) * 1000 * batch_data.shape[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        total_len = len(self.test_loader.dataset)
        test_loss /= total_len
        accuracy = correct_counts / total_len
        total_accuracy = np.sum(accuracy, axis=1) / config['pred_steps']
        if self._print_message:
            print(record_prediction)
            print(record_label)
            print(torch.round(torch.exp(record_prediction * (math.log(500) - math.log(1)) + math.log(1))))
            print(torch.round(torch.exp(record_label * (math.log(500) - math.log(1)) + math.log(1))))
            print(f"test loss：{test_loss:.6f}")
            print(accuracy)
            print(f"total accuracy {total_accuracy}")
        if self._save:
            total_params = sum(p.numel() for p in self.model.parameters())
            ave_runtime = times / total_len
            informations = config.get_all_values() + [total_params, ave_runtime, self.best_train_loss, test_loss,
                                                      total_accuracy]
            informations = pd.DataFrame([informations], columns=(list(config.get_all_keys()) +
                                                                 ['param_sum', 'ave_runtime', 'best_train_loss',
                                                                  'test_loss', 'test_accuracy']))
            self.test_params = pd.concat([self.test_params, informations], ignore_index=True)
            if self._print_message:
                print(self.test_params)
            if self._save:
                self.test_params.to_csv(os.path.join(self.result_path, "test_params.csv"), index=False)
                with open(os.path.join(self.result_path, "accuracy.txt"), 'w') as f:
                    f.write(str(accuracy))
        return test_loss, total_accuracy


# ===================== 主函数执行=====================
if __name__ == "__main__":
    test_losses = []
    test_accuracies = []
    intro = "test"
    for i in range(config['experiment_rounds']):
        trainer = ModelTrainer(config, intro)
        trainer.train()
        test_loss, test_accuracy = trainer.test()
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    for i in range(config['experiment_rounds']):
        print(f'experiment_round: {i} : test_loss: {test_losses[i]:6f}, test_accuracy: {test_accuracies[i]}')
    print(f'average : test_loss: {np.mean(test_losses)}, test_accuracy: {np.mean(test_accuracies)}')
    '''
    trainer = ModelTrainer(config)
    trainer.test("./results/Transformer/test_4/")
    '''
