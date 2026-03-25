import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from models.Transformer import Transformer
import pandas as pd

def init_results(config):
    result_path = os.path.join(config["result_path"], f'{config["model"]}/')
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
    result_path = os.path.join(result_path, f'test_{test_id}/')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    return full_test_results, test_id, result_path, test_result_path


class DistributedTrainer:
    def __init__(self, config, use_ddp=None):
        self.num_gpus = torch.cuda.device_count()
        if use_ddp is None:
            self.use_ddp = self.num_gpus > 1
        else:
            self.use_ddp = use_ddp and self.num_gpus > 1
        self.rank = 0
        self.world_size = 1
        if self.use_ddp:
            self._init_ddp()
        self.model = self._prepare_model(config)

    def _init_ddp(self):
        self.world_size = self.num_gpus
        self.rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=self.rank,
            world_size=self.world_size
        )
        torch.cuda.set_device(self.rank)

    def _prepare_model(self, config):
        """准备模型（单卡或多卡）"""
        if config['model'] == 'Transformer':
            model = Transformer(config).to(config['device'])
        else:
            print('invalid model')
            return None, None, None
        if self.use_ddp:
            model = model.cuda(self.rank)
            model = DDP(model, device_ids=[self.rank])
        else:
            model = model.cuda()
        return model

    def get_dataloader(self, dataset, batch_size, shuffle=True, drop_last=False):
        if self.use_ddp:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=drop_last
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
                pin_memory=True,
                drop_last=drop_last
            )

        return dataloader

    def get_device(self):
        """获取当前设备"""
        if self.use_ddp:
            return torch.device(f'cuda:{self.rank}')
        else:
            return torch.device('cuda')

    def save_model(self, path):
        """保存模型（自动处理DDP）"""
        if self.use_ddp:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        torch.save(state_dict, path)

    def load_model(self, path):
        """加载模型（自动处理DDP）"""
        state_dict = torch.load(path)
        if self.use_ddp:
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

    def is_main_process(self):
        """是否为主进程"""
        return self.rank == 0

    def cleanup(self):
        """清理资源"""
        if self.use_ddp and dist.is_initialized():
            dist.destroy_process_group()