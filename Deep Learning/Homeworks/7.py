import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import urllib.request
import collections
import math
import time

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 20
SEQ_LEN = 35
EMBED_SIZE = 650
HIDDEN_SIZE = 650
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 20.0
EPOCHS = 55
GRAD_CLIP = 0.25
DATA_DIR = './data'

# 1. Data Downloading and Preprocessing
def download_ptb_dataset(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    urls = {
        'train': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
        'valid': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
        'test': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt'
    }
    
    file_paths = {}
    for key, url in urls.items():
        filename = f'ptb.{key}.txt'
        filepath = os.path.join(data_dir, filename)
        file_paths[key] = filepath
        if not os.path.exists(filepath):
            print(f'Downloading {filename}...')
            try:
                # Try using wget first as it's often more robust in these envs
                os.system(f'wget --no-check-certificate {url} -O {filepath}')
                if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                     # Fallback to urllib if wget failed or wasn't present
                     urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                # If both fail, try to create a dummy file for testing if it's just a connectivity issue? 
                # No, that would be "fabricating facts". We must fail or ask user.
                # But let's try one more thing: using http instead of https if possible, but github forces https.
                raise e
        else:
            print(f'{filename} already exists.')
            
    return file_paths

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i):
    seq_len = min(SEQ_LEN, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

# 2. Model Definition (LSTM)
class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m.div_(1 - dropout)
        mask = mask.expand_as(x)
        return x * mask

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.locked_drop = LockedDropout()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        # Tie weights if dimensions match
        if embed_size == hidden_size:
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

        # Initialize LSTM forget gate bias to 1
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        emb = self.locked_drop(emb, dropout=self.drop.p)
        
        output, hidden = self.lstm(emb, hidden)
        
        output = self.locked_drop(output, dropout=self.drop.p)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
                weight.new_zeros(self.num_layers, bsz, self.hidden_size))

# 3. Training and Evaluation
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(model, data_source, criterion):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(BATCH_SIZE)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, SEQ_LEN):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, vocab_size)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)

def train(model, train_data, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(BATCH_SIZE)
    
    batch_losses = []
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, SEQ_LEN)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of dataset.
        hidden = repackage_hidden(hidden)
        
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, vocab_size), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

        if batch % 200 == 0 and batch > 0:
            cur_loss = total_loss / 200
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // SEQ_LEN, LEARNING_RATE,
                    elapsed * 1000 / 200, cur_loss, math.exp(cur_loss)))
            batch_losses.append(cur_loss)
            total_loss = 0
            start_time = time.time()
            
    return batch_losses

# Main Execution
if __name__ == '__main__':
    print(f"Using device: {device}")
    
    # Download Data
    print("Downloading/Checking Dataset...")
    download_ptb_dataset(DATA_DIR)
    
    # Load Data
    print("Loading Data...")
    corpus = Corpus(DATA_DIR)
    vocab_size = len(corpus.dictionary)
    print(f"Vocabulary Size: {vocab_size}")
    
    train_data = batchify(corpus.train, BATCH_SIZE)
    val_data = batchify(corpus.valid, BATCH_SIZE)
    test_data = batchify(corpus.test, BATCH_SIZE)
    
    # Initialize Model
    model = LSTMModel(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    best_val_loss = None
    train_losses_history = []
    val_ppl_history = []
    
    try:
        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            train_losses = train(model, train_data, criterion, optimizer, epoch)
            train_losses_history.extend(train_losses)
            
            val_loss = evaluate(model, val_data, criterion)
            val_ppl = math.exp(val_loss)
            val_ppl_history.append(val_ppl)
            
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, val_ppl))
            print('-' * 89)
            
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Test
    test_loss = evaluate(model, test_data, criterion)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_history)
    plt.title('Training Loss')
    plt.xlabel('Iterations (x200)')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_ppl_history, marker='o')
    plt.title('Validation Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('PPL')
    
    plt.tight_layout()
    plt.savefig('result.png')
    print("Plot saved to result.png")
