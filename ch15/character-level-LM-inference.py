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
input_text = 'The island had'
print(f'Starting string: "{input_text} ..."\n')
print("New text generate from model:")
print(sample(model, starting_str=input_text, in_device=mps_device, scale_factor=1.5))
# print(sample(model, starting_str='Navigating the sea', in_device=mps_device, scale_factor=0.5))
print('\n')