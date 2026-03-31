import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from models.Transformer import Transformer
from datetime import timedelta
import pandas as pd

def init_results(config):
    result_path = os.path.join(config["result_path"], f'{config["model"]}/')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    test_result_path = os.path.join(result_path, "test_results.csv")
    if not os.path.exists(test_result_path):
        full_test_results = pd.DataFrame(columns=(list(config.get_all_keys())+
                                                  ['param_sum','ave_runtime','best_train_loss','test_loss','test_accuracy']))
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
        print("initializing distributed trainer...")
        self.num_gpus = torch.cuda.device_count()
        if use_ddp is None:
            self.use_ddp = self.num_gpus > 1
        else:
            self.use_ddp = use_ddp and self.num_gpus > 1
        self.rank = 0
        self.world_size = 1
        if self.use_ddp:
            print("begin init ddp")
            self._init_ddp()
        self.model = self._prepare_model(config)


    def _init_ddp(self):
        '''
        self.world_size = self.num_gpus
        self.rank = int(os.environ.get('LOCAL_RANK', 0))
        store = dist.TCPStore(
            "localhost",
            8800,
            is_master=self.is_main_process(),
            timeout=timedelta(seconds=30),
            use_libuv=False
        )
        dist.init_process_group(
            backend='gloo',
            #init_method='env://',
            store=store,
            rank=self.rank,
            world_size=self.world_size
        )
        torch.cuda.set_device(self.rank)
        '''
        self.world_size = self.num_gpus
        self.rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=self.rank,
            world_size=self.world_size
        )
        dist.barrier()
        torch.cuda.set_device(self.rank)

    def _prepare_model(self, config):
        if config['model'] == 'Transformer':
            model = Transformer(config)
        else:
            print('invalid model')
            return None
        if self.use_ddp:
            if self.rank == 0:
                torch.save(model.state_dict(), "tmp.pth")
            dist.barrier()
            model.load_state_dict(torch.load("tmp.pth", map_location=self.get_device()))
            model = DDP(model, device_ids=[self.rank])
        model = model.to(self.get_device())

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
                pin_memory=True,
                drop_last=drop_last
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=True,
                drop_last=drop_last
            )

        return dataloader

    def get_device(self):
        if self.use_ddp:
            return torch.device(f'cuda:{self.rank}')
        else:
            return torch.device('cuda')

    def save_model(self, path):
        state_dict = self.model.state_dict()
        torch.save(state_dict, path)

    def load_model(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    def is_main_process(self):
        return self.rank == 0

    def cleanup(self):
        if self.use_ddp and dist.is_initialized():
            dist.destroy_process_group()
        if os.path.exists("tmp.pth"):
            os.remove("tmp.pth")

    def reduce_loss(self, loss, average=True):
        if self.world_size < 2:
            return loss
        with torch.no_grad():
            dist.all_reduce(loss)
            if average:
                loss = loss / self.world_size
            return loss
