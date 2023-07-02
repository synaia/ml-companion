import torch


def compute_z(a, b, c) -> torch.Tensor:
    r1 = torch.sub(a, b)
    r2 = torch.mul(r1, 2)
    z = torch.add(r2, c)
    return z


a = torch.tensor(1)
b = torch.tensor(4)
c = torch.tensor(2)

# compute scalars rank 0:
print(compute_z(a, b, c))

a = torch.tensor([1])
b = torch.tensor([4])
c = torch.tensor([2])

# compute scalars rank 1:
print(compute_z(a, b, c))

a = torch.tensor([[1]])
b = torch.tensor([[4]])
c = torch.tensor([[9]])

# compute scalars rank 2:
print(compute_z(a, b, c))