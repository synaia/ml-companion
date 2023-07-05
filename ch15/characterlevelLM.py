#######  Preparing the data
import numpy as np

with open('./downloaded/1268-0.txt', 'r', encoding='utf8') as fp:
    text = fp.read()

start_idx = text.find('THE MYSTERIOUS ISLAND')
end_idx = text.find('End of the Project Gutenberg')
text = text[start_idx:end_idx]
char_set = set(text)
print('Total length: ', len(text))
print('Unique characters: ', len(char_set))

# building the dictionary
chars_sorted = sorted(char_set)
char2int = {ch:i for i, ch in enumerate(chars_sorted)}

char_array = np.array(chars_sorted)
text_encoded = np.array([char2int[ch] for ch in text], dtype=np.int32)
print('Text encoded shape: ', text_encoded.shape)
print(text[:15], '== Encoding ==>', text_encoded[:15])
print(text_encoded[15:21], '== Reverse ==>', ''.join(char_array[text_encoded[15:21]]))

# for ex in text_encoded[:10]:
#     print(f'{ex} -> {char_array[ex]}')

# self-defined Dataset class
from torch.utils.data import Dataset
import torch
import numpy as np

mps_device = torch.device("mps")


class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self, idx):
        text_chunk = self.text_chunks[idx]
        # wtf?
        return text_chunk[: -1].long(), text_chunk[1:].long()


seq_length = 40
chunk_size = seq_length + 1
text_chunks = [text_encoded[i:i+chunk_size] for i in range(len(text_encoded) - chunk_size)]
seq_dataset = TextDataset(torch.tensor(np.array(text_chunks)).to(mps_device))

# examples.
# for i, (seq, target) in enumerate(seq_dataset):
#     print('  Input (x):', repr(''.join(char_array[seq])))
#     print('  Target (x):', repr(''.join(char_array[target])))
#     print()
#     if i == 1:
#         break

# praparing the dataset
from torch.utils.data import DataLoader
batch_size = 64
torch.manual_seed(1)
seq_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

#######  Building the RNN model
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, device=mps_device)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)
        # self.logsoftmax = nn.LogSoftmax(1)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        # out = self.logsoftmax(x)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden.to(mps_device), cell.to(mps_device)


vocab_size = len(char_array)
embed_dim = 256
rnn_hidden_size = 512
torch.manual_seed(1)
model = RNN(vocab_size, embed_dim, rnn_hidden_size)
model = model.to(mps_device)
print(model)


## save best model utility
class SaveBestModel:
    def __init__(self):
        self.best_valid_loss = float('inf')

    def __call__(self, current_valid_loss, epoch, model, optimizer, loss_criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_criterion,
            },
            'outputs/best_model.pth')


#######  Performing next-character prediction and sampling
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # or 0.005

num_epochs = 10_000
torch.manual_seed(1)
save_best = SaveBestModel()
# for epoch in range(num_epochs):
#     hidden, cell = model.init_hidden(batch_size=batch_size)
#     # hidden.to(mps_device)
#     # cell.to(mps_device)
#     seq_batch, target_batch = next(iter(seq_dl))
#     seq_batch = seq_batch.to(mps_device)
#     target_batch = target_batch.to(mps_device)
#     optimizer.zero_grad()
#     loss = 0
#     for c in range(seq_length):
#         pred, hidden, cell = model(seq_batch[:, c], hidden, cell)
#         hidden.to(mps_device)
#         cell.to(mps_device)
#         loss += loss_fn(pred, target_batch[:, c])
#     loss.backward()
#     optimizer.step()
#     loss = loss.item() / seq_length
#     # current_valid_loss, epoch, model, optimizer, loss_criterion):
#     save_best(loss, epoch, model, optimizer, loss_fn)
#     print(f'Epoch {epoch}  loss {loss:.4f}')



# Evaluation #
from torch.distributions.categorical import Categorical


def sample(model: RNN, starting_str, len_generated_text=500, scale_factor=1., in_device=None):
    encoded_input = torch.tensor(
        [char2int[s] for s in starting_str],
        device=in_device
    )
    encoded_input = torch.reshape(
        encoded_input, (1, -1),
    )
    generated_str = starting_str

    model.eval()
    hidden, cell = model.init_hidden(1)
    hidden.to(in_device)
    cell.to(in_device)
    for c in range(len(starting_str) - 1):
        _, hidden, cell = model( # check this
            encoded_input[:, c].view(1), hidden, cell
        )

    last_char = encoded_input[:, -1]
    for i in range(len_generated_text):
        logits, hidden, cell = model(
            last_char.view(1), hidden, cell
        )
        logits = torch.squeeze(logits, 0) # wtf?
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample()
        generated_str += str(char_array[last_char])

    return generated_str


# loading the saved model.
model = RNN(vocab_size, embed_dim, rnn_hidden_size)
checkpoint = torch.load('./outputs/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.to(mps_device)

# Experiment:
# print('\n\nExperiment whit mps\n')
torch.manual_seed(1)
# print(sample(model, starting_str='The island had', in_device=mps_device))


