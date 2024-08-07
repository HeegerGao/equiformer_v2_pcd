import torch
from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
import time 

data = {
    'atomic_numbers': torch.tensor([6, 8, 1, 6, 8, 1, 1, 1, 6, 8], dtype=torch.long),  # 所有分子系统中的原子种类
    'pos': torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],  # 第一个分子系统的坐标
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0],  # 第二个分子系统的坐标
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]  # 第三个分子系统的坐标
    ], dtype=torch.float),
    'batch': torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2], dtype=torch.long),  # 标识每个原子所属的分子系统
    'edge_index': torch.tensor([
        [0, 0, 1, 3, 3, 4, 4, 5, 5, 6, 8], 
        [1, 2, 2, 4, 5, 5, 6, 6, 7, 7, 9]
    ], dtype=torch.long),  # 边的连接关系
    'edge_distance': torch.tensor([
        1.0, 1.0, 1.414, 
        1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.0, 
        1.0
    ], dtype=torch.float),  # 边的距离
    'edge_distance_vec': torch.tensor([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 
        [1.0, 0.0, 0.0]
    ], dtype=torch.float),  # 边的向量
    'natoms': torch.tensor([3, 5, 2], dtype=torch.long)  # 每个分子系统中的原子数量
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
net = EquiformerV2_OC20(
    use_pbc=False,
    num_atoms=0,      # not used
    bond_feat_dim=0,  # not used
    num_targets=0,    # not used
    max_radius=2.0,
    max_neighbors=2,
    num_layers=4,
).to(device)

# 记录开始时间
start_time = time.time()

# Forward pass
output = net(data)
print(output)
# 记录结束时间
end_time = time.time()

# 计算用时
elapsed_time = end_time - start_time

print(f"程序运行时间: {elapsed_time:.6f} 秒")