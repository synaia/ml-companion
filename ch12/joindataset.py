import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class JointDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


torch.manual_seed(1)
t_x = torch.rand([4, 3], dtype=torch.float32)
t_y = torch.rand(4)
joint_dataset = JointDataset(t_x, t_y)
for example in joint_dataset:
    print(f'x: {example[0]}, y: {example[1]}')
print('---------------------------------------------------\n')

for epoch in range(2):
    print(f'epoch {epoch + 1}')
    data_loader = DataLoader(dataset=joint_dataset, batch_size=2, shuffle=True)
    for i, batch in enumerate(data_loader, 1):
        print(f'batch {i}: x: {batch[0]}, y: {batch[1]}')



