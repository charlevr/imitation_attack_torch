from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator

import requests
import io
import pandas as pd
import csv
import tqdm

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.autograd import Variable

class BatchWrapper:
      def __init__(self, dl, x_var, y_vars):
            self.dl, self.x_var, self.y_vars = dl, x_var, y_vars # we pass in the list of attributes for x 

      def __iter__(self):
            for batch in self.dl:
                  x = getattr(batch, self.x_var) # we assume only one input in this wrapper

                  if self.y_vars is None: # we will concatenate y into a single tensor
                        y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
                  else:
                        y = torch.zeros((1))

                  yield (x, y)

      def __len__(self):
            return len(self.dl)

tokenize = lambda x:x.split()

TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = Field(sequential=False, use_vocab=False)

urllist = ['https://raw.githubusercontent.com/charlevr/imitation_attack/master/imitator_samples.csv',
          'https://raw.githubusercontent.com/charlevr/imitation_attack/master/original_authors.csv',
	  'https://raw.githubusercontent.com/charlevr/imitation_attack/master/imitations.csv']

f_list = [requests.get(url).content.decode('utf-8') for url in urllist]

df_imit_samples = pd.read_csv(io.StringIO(f_list[0]))
df_imit = pd.read_csv(io.StringIO(f_list[2]))
df_orig_samples = pd.read_csv(io.StringIO(f_list[1]))

def get_orig_rows(author):
	rows = []
	for i,row in df_orig_samples[df_orig_samples['creator'] == author].iterrows():
		rows.append([list(row)[1], 0])
	return rows
					  
def get_attacks_on_author(df, intended_author):
	rows = []
	for i,row in df.iterrows():
		r = list(row)
		if r[3] == intended_author:
			rows.append([r[1], 1])
	return rows


train_set = pd.DataFrame(get_orig_rows('participant 1') + \
get_orig_rows('participant 2') + \
get_attacks_on_author(df_imit[:134], 'participant 1') + \
get_attacks_on_author(df_imit[:134], 'participant 2')).sample(frac = 1)

test_set = pd.DataFrame(get_orig_rows('participant 3') + \
get_attacks_on_author(df_imit[134:], 'participant 3')).sample(frac = 1)

test_set.columns = train_set.columns = ['msg', 'class']

train_set.to_csv(path_or_buf=r'./data/train.csv', index=False)
test_set.to_csv(path_or_buf=r'./data/test.csv', index=False)


tv_datafields = [("msg", TEXT), ("class", LABEL)]

trn, vld = TabularDataset.splits(path="data", train='train.csv', validation='test.csv', format='csv', skip_header=True, fields=tv_datafields)

TEXT.build_vocab(trn)

train_iter, val_iter = BucketIterator.splits( (trn, vld), batch_sizes=(64, 64), device=-1, sort_key=lambda x:len(x.comment_text), sort_within_batch=False, repeat=False)

train_dl = BatchWrapper(train_iter, "msg", ["class"])
valid_dl = BatchWrapper(val_iter, "msg", ["class"])

class SimpleLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim, emb_dim=300, num_linear=1):
        super().__init__() # don't forget to call this!
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1)
        self.linear_layers = []
        for _ in range(num_linear):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, 6)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
            preds = self.predictor(feature)
        return preds
 

em_sz = 100 
nh = 500 
nl = 3 
model = SimpleLSTMBaseline(nh, emb_dim=em_sz)#, num_linear=nl)
  
opt = optim.Adam(model.parameters(), lr=1e-2)
loss_func = nn.BCEWithLogitsLoss()

epochs = 2

for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_corrects = 0
    model.train() # turn on training mode
    for x, y in tqdm.tqdm(train_dl): # thanks to our wrapper, we can intuitively iterate over our data!
        opt.zero_grad()

        preds = model(x)
        print(y)
        print(preds)
        loss = loss_func(y, preds)
        loss.backward()
        opt.step()

        running_loss += loss.data[0] * x.size(0)

    epoch_loss = running_loss / len(trn)

    # calculate the validation loss for this epoch
    val_loss = 0.0
    model.eval() # turn on evaluation mode
    for x, y in valid_dl:
        preds = model(x)
        loss = loss_func(y, preds)
        val_loss += loss.data[0] * x.size(0)

    val_loss /= len(vld)
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
