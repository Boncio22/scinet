#   Copyright 2018 SciNet (https://github.com/eth-nn-physics/nn_physical_concepts)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import pickle
import gzip
import numpy as np

from torch.utils.data import Dataset, DataLoader


def load(validation_size_p, file_name):
    """
    Params:
    validation_size_p: percentage of data to be used for validation
    file_name (str): File containing the data
    """
    f = gzip.open(file_name + ".plk.gz", 'rb')
    data, states, params = pickle.load(f)
    states = np.array(states)
    train_val_separation = int(len(data[0]) * (1 - validation_size_p))
    training_data = [data[i][:train_val_separation] for i in [0, 1, 2]]
    training_states = states[:train_val_separation]
    validation_data = [data[i][train_val_separation:] for i in [0, 1, 2]]
    validation_states = states[train_val_separation:]
    f.close()
    return (training_data, validation_data, training_states, validation_states, params)

class SciNetDataset(Dataset):
    """docstring for SciNetDataset"""
    def __init__(self, data):
        super(SciNetDataset, self).__init__()
        x, t, label = data

        self.x = x.astype(np.float32)
        self.t = t.astype(np.float32)
        self.label = label.astype(np.float32)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.label[idx]

def get_data(dataset: str='oscillator', batch_size: int=512):
    training_data, validation_data, _, _, _ = load(0.05, f'{dataset}_data')
    train_data = SciNetDataset(training_data)
    valid_data = SciNetDataset(validation_data)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

    return train_loader, valid_loader

if __name__ == '__main__':
    train_loader, valid_loader = get_data()

    print(len(train_loader))