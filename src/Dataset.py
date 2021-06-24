import torch
from torch.utils.data import Dataset
import os
import csv


class Wiki_v2_Dataset(Dataset):
    _PATHS = {'train': 'lan0_out_train.csv', 'test': 'lan0_out_test.csv', 'dev': 'lan0_out_dev.csv'}

    def __init__(self, dir, type='train', salient_features=False):
        super().__init__()
        self._dataset = []
        self._salient_features = salient_features
        with open(os.path.join(dir, self._PATHS[type])) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t',
                                    quoting=csv.QUOTE_MINIMAL)
            self._dataset.extend([line for line in csv_reader])

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        # 358bd9e861," Sons of ****, why couldn`t they put them on the
        # releases we already bought","Sons of ****,",negative
        result = [self._dataset[item][0][-1], self._dataset[item][0][:-1]]
        if self._salient_features:
            result.append(self._dataset[item][3])

        # print(tuple(result))
        return tuple(result)

class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)