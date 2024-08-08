import torch
from nets.equiformer_v2.equiformer_v2_pcd import EquiformerV2_PCD
import time
from utils import test_equivariance

data = {
    "pos": torch.tensor([     # random 20 points from 3 graphs
        [0.4890, 0.9572, 0.1684],
        [0.2664, 0.6796, 0.1454],
        [0.8225, 0.7862, 0.0442],
        [0.2148, 0.2654, 0.6986],
        [0.5122, 0.3097, 0.2329],
        [0.2652, 0.0590, 0.8858],
        [1.4890, 1.9572, 1.1684],
        [1.2664, 1.6796, 1.1454],
        [1.8225, 1.7862, 1.0442],
        [1.2148, 1.2654, 1.6986],
        [1.5122, 1.3097, 1.2329],
        [1.2652, 1.0590, 1.8858],
        [0.4567, 0.7652, 0.0201],
        [0.7795, 0.8571, 0.7655],
        [0.6167, 0.9216, 0.3828],
        [0.9317, 0.8438, 0.5071],
        [0.4775, 0.2155, 0.2570],
        [0.8321, 0.4023, 0.8741],
        [0.2145, 0.6816, 0.4394],
        [0.1473, 0.9960, 0.8025]], dtype=torch.float),
    "input_types": [0, 1],
    "input_channels": [3, 1],
    "feature":  torch.tensor([ # random features for 20 points：RGB + 1 type-1 feature
        [0.8378, 0.9013, 0.5233, 0.2853, 0.3983, 0.6622],
        [0.3812, 0.7911, 0.3282, 0.4537, 0.2018, 0.1417],
        [0.6450, 0.3563, 0.2380, 0.1900, 0.6948, 0.1455],
        [0.3276, 0.6768, 0.2650, 0.5831, 0.7972, 0.0154],
        [0.9821, 0.7187, 0.9121, 0.6613, 0.7160, 0.2816],
        [0.3828, 0.0957, 0.6754, 0.2865, 0.5819, 0.6195],
        [0.8378, 0.9013, 0.5233, 0.2853, 0.3983, 0.6622],
        [0.3812, 0.7911, 0.3282, 0.4537, 0.2018, 0.1417],
        [0.6450, 0.3563, 0.2380, 0.1900, 0.6948, 0.1455],
        [0.3276, 0.6768, 0.2650, 0.5831, 0.7972, 0.0154],
        [0.9821, 0.7187, 0.9121, 0.6613, 0.7160, 0.2816],
        [0.3828, 0.0957, 0.6754, 0.2865, 0.5819, 0.6195],
        [0.1316, 0.5390, 0.2258, 0.2251, 0.7644, 0.2659],
        [0.8795, 0.2098, 0.1461, 0.5827, 0.4320, 0.7263],
        [0.6588, 0.5016, 0.5694, 0.8368, 0.3944, 0.4388],
        [0.3906, 0.7490, 0.6863, 0.0520, 0.3927, 0.3871],
        [0.6148, 0.1303, 0.9124, 0.2134, 0.2048, 0.7852],
        [0.4781, 0.0651, 0.7340, 0.9224, 0.5175, 0.8874],
        [0.4266, 0.7268, 0.8412, 0.7313, 0.9292, 0.0203],
        [0.3204, 0.6804, 0.6473, 0.9545, 0.4128, 0.5965]]),
    'batch': torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], dtype=torch.long),  # the index for each point
    'graph_pcd_num': torch.tensor([6, 6, 8], dtype=torch.long),  # point number for each graph
}

# Wrap in a class to mimic data object
class Data:
    def __init__(self, data_dict, device="cuda"):
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.to(device)
            setattr(self, key, value)


device = "cuda:7"
data = Data(data, device=device)
net = EquiformerV2_PCD(input_types=data.input_types, input_channels=data.input_channels, num_layers=4)
net = net.to(device)
print(net.num_params)
start_time = time.time()

# Forward pass
heatmap, orientation = net(data)
print(heatmap.shape, orientation.shape)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"time: {elapsed_time:.6f} seconds")


test_equivariance(net, data)