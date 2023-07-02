import torch
from torch import nn


### simple procedure of computing gradient of the loss with respect to trainable variables.
# z = wx + b, Loss = (y - z)^2
w = torch.tensor(1., requires_grad=True)
b = torch.tensor(.5, requires_grad=True)
x = torch.tensor([1.4])
y = torch.tensor([2.1])
z = torch.add(torch.mul(w, x), b)
loss = (y - z).pow(2).sum()
loss.backward()

# ∂Loss/∂w = 2x(wx + b - y)  the partial derivative of Loss with respect to w.
print(f"∂Loss/∂w : {w.grad}")
print(f"∂Loss/∂b : {b.grad}")



# simple fully connected Sequential demo

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU()
)

print(model)

nn.init.xavier_uniform_(model[0].weight)
l1_weight = 0.01
l1_penalty = l1_weight * model[2].weight.abs().sum()


