import torch
from characterlevelLM import RNN, sample


# loading the saved model.
model = RNN(80, 256, 512)
checkpoint = torch.load('./outputs/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

mps_device = torch.device("mps")
model.to(mps_device)

# Experiment:
print('\n\n\n\n\nPerforming next-character prediction and sampling (using mps-gpu device)\n')
torch.manual_seed(1)
print(sample(model, starting_str='The island had', in_device=mps_device))
# print(sample(model, starting_str='Captain Grant said', in_device=mps_device))
print('\n\n\n\n\n\n\n\n\n')