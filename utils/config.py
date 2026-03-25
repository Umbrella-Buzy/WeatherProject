import yaml
import torch

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.params = yaml.safe_load(file)
        self.model = self.params['Global']['model']

    def __getitem__(self, key):
        if key == 'device' and self.params['Global'][key] == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        if key in self.params[self.model]:
            return self.params[self.model][key]
        else:
            return self.params['Global'][key]

    def __setitem__(self, key, value):
        self.params[self.model][key] = value
        self.__setattr__(key, value)

    def __contains__(self, key):
        return key in self.params[self.model]

    def get_all_keys(self):
        return (list(self.params['Global'].keys()) + list(self.params[self.model].keys()))

    def get_all_values(self):
        return [self[key] for key in self.get_all_keys()]