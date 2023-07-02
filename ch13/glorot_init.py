import torch
import torch.nn as nn

torch.manual_seed(1)
w = torch.empty(2, 3)
# nn.init.xavier_normal(w)
print(w)

# x_method_() NO making copy of the input ... Whats imput?
# nn.init.xavier_normal_(w)
print(w)


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.empty(2, 3, requires_grad=True)
        nn.init.xavier_normal_(self.w1)
        self.w2 = torch.empty(1, 2, requires_grad=True)
        nn.init.xavier_normal_(self.w2)

mymodule = MyModule()
print(mymodule.w1, mymodule.w2)