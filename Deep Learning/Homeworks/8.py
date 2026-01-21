import sys
import os
import typing
import collections

# ==========================================
# 0. Environment Patch (Fix for Python 3.7.x)
# ==========================================
# Fix ImportError: cannot import name 'OrderedDict' from 'typing'
if sys.version_info[:3] < (3, 7, 2):
    if not hasattr(typing, 'OrderedDict'):
        typing.OrderedDict = collections.OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import random
import math
import time
import nltk
from collections import Counter
import zipfile
import urllib.request
import urllib.parse
import shutil
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# Ensure nltk resources are available
def check_nltk_resource(resource_name):
    try:
        nltk.data.find(f'tokenizers/{resource_name}')
    except (LookupError, zipfile.BadZipFile, OSError):
        print(f"Downloading or Re-downloading NLTK resource: {resource_name}...")
        try:
            nltk.download(resource_name)
        except Exception as e:
            print(f"Error downloading {resource_name}: {e}")

check_nltk_resource('punkt')
check_nltk_resource('punkt_tab')

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # Paths
    DATA_DIR = './data/flickr8k'
    IMG_DIR = os.path.join(DATA_DIR, 'images')
    ANN_DIR = os.path.join(DATA_DIR, 'annotations')
    # Flickr8k specific paths
    CAPTION_FILE = os.path.join(ANN_DIR, 'Flickr8k.token.txt')
    CHECKPOINT_DIR = './checkpoints'
    
    # URLs (Flickr8k)
    # Using GitHub release links which are generally reliable
    URL_IMAGES = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    URL_TEXT = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

    # Hyperparameters
    embed_size = 512
    hidden_size = 512
    num_heads = 8
    num_layers = 4
    dropout = 0.3 # Increased to 0.3 as requested
    vocab_size = 0 # Will be set after building vocab
    max_seq_len = 30
    
    batch_size = 32 
    learning_rate = 1e-4 # Decreased from 4e-4 to fix fluctuation
    num_epochs = 20 # 10 is too short for lower LR, set to 20
    freq_threshold = 5
    
    # Use GPU if available
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    
    # Debug/Demo mode
    USE_DUMMY_DATA = False 
    
    USE_MIRROR = False
    MIRROR_MAP = {}

config = Config()

# ==========================================
# 2. Data Download & Preparation
# ==========================================
def download_file(url, dest_path):
    if os.path.exists(dest_path):
        # Check for corrupted zip files
        if dest_path.endswith('.zip'):
            try:
                with zipfile.ZipFile(dest_path, 'r') as z:
                    pass # Just check if we can open it
                print(f"File already exists: {dest_path}")
                return
            except zipfile.BadZipFile:
                print(f"Found corrupted zip file: {dest_path}. Deleting and re-downloading...")
                os.remove(dest_path)
        else:
            print(f"File already exists: {dest_path}")
            return

    # Build candidate URLs: primary first, then mirrors (if configured)
    candidate_urls = [url]
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc
        if config.USE_MIRROR and isinstance(config.MIRROR_MAP, dict):
            mirror = config.MIRROR_MAP.get(host)
            if mirror:
                # Mirror may be a full URL (with scheme) or just a host
                if mirror.startswith('http://') or mirror.startswith('https://'):
                    mirror_url = mirror.rstrip('/') + parsed.path
                else:
                    mirror_url = parsed.scheme + '://' + mirror.rstrip('/') + parsed.path
                # Put mirror first to try it before original
                candidate_urls.insert(0, mirror_url)
    except Exception:
        # If parsing fails, proceed with original URL
        pass

    last_exc = None
    for candidate in candidate_urls:
        print(f"Attempting download: {candidate} -> {dest_path}")
        
        # 1. Try aria2c (fastest, multi-connection)
        if shutil.which('aria2c'):
            print("Using aria2c...")
            ret = os.system(f"aria2c -x 4 -s 4 -c -o '{os.path.basename(dest_path)}' '{candidate}' --dir='{os.path.dirname(dest_path) or '.'}'")
            if ret == 0:
                print("Download complete (aria2c).")
                return
            else:
                print(f"aria2c failed with code {ret}, trying next tool...")

        # 2. Try wget (reliable, resume support)
        if shutil.which('wget'):
            print("Using wget...")
            ret = os.system(f"wget -c '{candidate}' -O '{dest_path}'")
            if ret == 0:
                print("Download complete (wget).")
                return
            else:
                print(f"wget failed with code {ret}, trying next tool...")

        # 3. Fallback to urllib (basic)
        try:
            print("Using urllib...")
            urllib.request.urlretrieve(candidate, dest_path)
            print("Download complete (urllib).")
            return
        except Exception as e:
            print(f"Download failed for {candidate}: {e}")
            last_exc = e

    # If we reach here, all candidates failed
    raise RuntimeError(f"Failed to download {url} -> {dest_path}") from last_exc

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Unzip complete.")

def prepare_dataset():
    if config.USE_DUMMY_DATA:
        print("Using Dummy Data. Skipping download.")
        return

    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
    if not os.path.exists(config.IMG_DIR):
        os.makedirs(config.IMG_DIR)
    if not os.path.exists(config.ANN_DIR):
        os.makedirs(config.ANN_DIR)
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)

    # 1. Annotations (Text)
    if not os.path.exists(config.CAPTION_FILE):
        text_zip = os.path.join(config.DATA_DIR, "Flickr8k_text.zip")
        download_file(config.URL_TEXT, text_zip)
        unzip_file(text_zip, config.ANN_DIR)

    # 2. Images
    # Check if images folder is populated (Flickr8k has 8091 images)
    if len(os.listdir(config.IMG_DIR)) < 8000:
        img_zip = os.path.join(config.DATA_DIR, "Flickr8k_Dataset.zip")
        download_file(config.URL_IMAGES, img_zip)
        # Extract to DATA_DIR first
        unzip_file(img_zip, config.DATA_DIR)
        
        # The zip usually contains a folder named 'Flicker8k_Dataset' (note the 'er')
        extracted_folder = os.path.join(config.DATA_DIR, "Flicker8k_Dataset")
        if os.path.exists(extracted_folder):
            print(f"Moving images from {extracted_folder} to {config.IMG_DIR}...")
            for f in os.listdir(extracted_folder):
                shutil.move(os.path.join(extracted_folder, f), config.IMG_DIR)
            os.rmdir(extracted_folder)

    print("Dataset preparation check passed.")

# ==========================================
# 3. Vocabulary & Dataset
# ==========================================
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return nltk.tokenize.word_tokenize(text.lower())

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, vocab=None, freq_threshold=5):
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab
        self.captions_file = captions_file
        
        # Load captions
        self.captions = [] # List of (image_name, caption_text)
        print(f"Loading annotations from {captions_file}...")
        with open(captions_file, "r") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                img_id_map = parts[0] # e.g., 1000268201_693b08cb0e.jpg#0
                caption_text = parts[1]
                img_name = img_id_map.split('#')[0]
                self.captions.append((img_name, caption_text))
        
        if self.vocab is None:
            print("Building vocabulary...")
            self.vocab = Vocabulary(freq_threshold)
            texts = [cap[1] for cap in self.captions]
            self.vocab.build_vocabulary(texts)
            print(f"Vocabulary built with {len(self.vocab)} tokens.")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        img_name, caption = self.captions[index]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Fallback for missing images
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)
            
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return image, torch.tensor(numericalized_caption), img_name

class DummyDataset(Dataset):
    def __init__(self, transform=None, vocab=None, freq_threshold=5):
        self.transform = transform
        self.vocab = vocab
        self.length = 100 # Dummy size
        
        # Dummy captions
        self.captions = [
            "a cat sitting on a table",
            "a dog running in the park",
            "a man riding a bike",
            "a woman holding an umbrella",
            "a group of people standing in a room"
        ] * 20
        
        if self.vocab is None:
            print("Building vocabulary (Dummy)...")
            self.vocab = Vocabulary(freq_threshold=1) # Low threshold for dummy
            self.vocab.build_vocabulary(self.captions)
            print(f"Vocabulary built with {len(self.vocab)} tokens.")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        caption = self.captions[index]
        # Dummy image (random noise)
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        if self.transform:
            image = self.transform(image)
            
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return image, torch.tensor(numericalized_caption), f"dummy_{index}"

class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        img_names = [item[2] for item in batch]
        return imgs, targets, img_names

# ==========================================
# 4. Model: CNN-Transformer
# ==========================================
class CNNEncoder(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(CNNEncoder, self).__init__()
        # Load pretrained ResNet101 (Better feature extraction)
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Conv2d(2048, embed_size, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_size)
        
        # Learnable positional encoding for 7x7 feature map
        self.pos_embed = nn.Parameter(torch.randn(49, 1, embed_size))

        # Freeze ResNet weights to prevent destroying pretrained features
        if not train_CNN:
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            # If train_CNN is True, we only unfreeze the last few layers (e.g., layer4)
            # to allow fine-tuning without destroying early features.
            for name, param in self.resnet.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, images):
        features = self.resnet(images)          # [Batch, 2048, 7, 7]
        features = self.embed(features)         # [Batch, embed_size, 7, 7]
        features = self.bn(features)
        features = features.flatten(2)
        features = features.permute(2, 0, 1)    # [49, Batch, embed_size]
        
        # Add positional encoding
        features = features + self.pos_embed
        
        return features

    def train(self, mode=True):
        super(CNNEncoder, self).train(mode)
        # Always freeze ResNet batchnorm to avoid destroying pretrained features
        self.resnet.eval()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerDecoderModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_heads, num_layers, dropout):
        super(TransformerDecoderModel, self).__init__()
        self.pos_encoder = PositionalEncoding(embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, 
                                                 dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions, tgt_padding_mask=None):
        # Scale embedding by sqrt(d_model) as per Attention Is All You Need
        tgt = self.embedding(captions) * math.sqrt(self.embedding.embedding_dim)
        tgt = tgt.permute(1, 0, 2) # [Seq_Len, Batch, Embed]
        tgt = self.pos_encoder(tgt)
        tgt = self.dropout(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(config.device)
        
        output = self.transformer_decoder(tgt, features, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        output = self.fc_out(output) # [Seq_Len, Batch, Vocab]
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class ImageCaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_heads, num_layers, dropout, pad_idx, train_CNN=False):
        super(ImageCaptionModel, self).__init__()
        self.pad_idx = pad_idx
        self.encoder = CNNEncoder(embed_size, train_CNN=train_CNN)
        self.decoder = TransformerDecoderModel(embed_size, hidden_size, vocab_size, num_heads, num_layers, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        # Create padding mask for the decoder
        tgt_padding_mask = (captions == self.pad_idx)
        outputs = self.decoder(features, captions, tgt_padding_mask)
        return outputs

    def caption_image(self, image, vocab, max_len=20):
        self.eval()
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0)) # [49, 1, Embed]
            start_token = vocab.stoi["<SOS>"]
            tgt_indexes = [start_token]
            for _ in range(max_len):
                tgt_tensor = torch.LongTensor(tgt_indexes).unsqueeze(1).to(config.device)
                # No padding mask needed for inference as batch size is 1 and no padding
                output = self.decoder(features, tgt_tensor.transpose(0, 1))
                last_token_logits = output[-1, 0, :]
                predicted_token = last_token_logits.argmax(0).item()
                tgt_indexes.append(predicted_token)
                if predicted_token == vocab.stoi["<EOS>"]:
                    break
        return [vocab.itos[i] for i in tgt_indexes]

    def caption_image_beam_search(self, image, vocab, beam_size=5, max_len=20, alpha=0.7):
        self.eval()
        k = beam_size
        with torch.no_grad():
            # Encoder
            features = self.encoder(image.unsqueeze(0)) # [49, 1, Embed]
            # Expand features for beam size
            features = features.expand(-1, k, -1) # [49, k, Embed]
            
            # Tensor to store top k sequences; shape [k, 1]
            seqs = torch.LongTensor([[vocab.stoi["<SOS>"]]] * k).to(config.device) # [k, 1]
            
            # Tensor to store top k scores; shape [k]
            top_k_scores = torch.zeros(k).to(config.device)
            
            # Completed sequences
            complete_seqs = []
            complete_seqs_scores = []
            
            step = 1
            while True:
                # Decoder forward
                # seqs: [k, seq_len] (batch-first). The decoder expects captions as [Batch, Seq_len].
                tgt_tensor = seqs  # [k, seq_len]

                output = self.decoder(features, tgt_tensor) # [seq_len, k, vocab_size]
                last_token_logits = output[-1, :, :] # [k, vocab_size]
                
                scores = torch.log_softmax(last_token_logits, dim=1) # [k, vocab_size]
                
                # Add previous scores
                scores = top_k_scores.unsqueeze(1).expand_as(scores) + scores
                
                # For the first step, all k points are the same <SOS>, so we only look at the first one
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
                else:
                    # Flatten scores to find top k across all (prev_beam, word) pairs
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
                
                # Convert flattened indices to (prev_beam_idx, word_idx)
                prev_beam_idxs = top_k_words // len(vocab)
                next_word_idxs = top_k_words % len(vocab)
                
                # Create new sequences
                seqs_new = torch.cat([seqs[prev_beam_idxs], next_word_idxs.unsqueeze(1)], dim=1)
                
                # Check for EOS
                incomplete_inds = []
                for i, word_idx in enumerate(next_word_idxs):
                    if word_idx == vocab.stoi["<EOS>"]:
                        # Length normalization: score / (len^alpha)
                        length = seqs_new[i].size(0)
                        norm_score = top_k_scores[i].item() / (length ** alpha)
                        complete_seqs.append(seqs_new[i].tolist())
                        complete_seqs_scores.append(norm_score)
                    else:
                        incomplete_inds.append(i)
                
                # If we don't have enough incomplete sequences, stop
                if len(incomplete_inds) == 0:
                    break
                    
                # Update sequences and scores for next step
                seqs = seqs_new[incomplete_inds]
                top_k_scores = top_k_scores[incomplete_inds]
                features = features[:, :len(incomplete_inds), :] # Adjust features batch size
                
                # If we have enough completed sequences, we can stop or continue to find better ones
                # Here we stop if we have at least k completed, or if max_len reached
                if len(complete_seqs) >= k:
                    break
                    
                if step >= max_len:
                    # Add remaining incomplete sequences
                    for i in range(len(seqs)):
                        length = seqs[i].size(0)
                        norm_score = top_k_scores[i].item() / (length ** alpha)
                        complete_seqs.append(seqs[i].tolist())
                        complete_seqs_scores.append(norm_score)
                    break
                
                step += 1
                k = len(seqs) # Update k for next iteration (it might shrink)

            # Choose the sequence with the highest score
            if len(complete_seqs) > 0:
                i = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[i]
            else:
                seq = seqs[0].tolist()
                
            return [vocab.itos[idx] for idx in seq]

# ==========================================
# 5. Training & Evaluation Functions
# ==========================================
def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    print(f"Epoch {epoch+1} Training...")
    
    for idx, (imgs, captions, _) in enumerate(loader):
        imgs = imgs.to(config.device)
        captions = captions.to(config.device)
        
        optimizer.zero_grad()
        outputs = model(imgs, captions[:, :-1])
        targets = captions[:, 1:]
        
        # Outputs: [Seq_Len, Batch, Vocab] -> [Batch, Seq_Len, Vocab]
        outputs = outputs.permute(1, 0, 2)
        
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if idx % 10 == 0: # More frequent logging for dummy
            print(f"Step [{idx}/{len(loader)}] Loss: {loss.item():.4f}")
            
    return total_loss / len(loader)

def evaluate_metrics(model, loader, vocab):
    model.eval()
    
    # Prepare data for pycocoevalcap
    # gts: {image_id: [{'caption': 'text'}, ...]}
    # res: [{image_id: image_id, 'caption': 'text'}]
    gts = {}
    res = {} # Use dict for faster lookup, convert to list later
    
    # Pre-load all ground truth captions for the dataset to ensure we have all 5 references
    # even if the loader returns them one by one.
    # We can access the underlying dataset from the loader
    if hasattr(loader.dataset, 'subset'):
        full_ds = loader.dataset.subset.dataset
        # But we only want the ones in the current split (val or test)
        # The loader iterates over the subset.
        # We can build the GT dict on the fly, but we must ensure we collect ALL references for each image.
        # Since our dataset returns (img, caption, img_name), and the same img_name appears 5 times
        # with different captions, iterating through the whole loader will naturally collect all 5.
        pass
    
    print("Generating captions for evaluation...")
    with torch.no_grad():
        for idx, (imgs, captions, img_names) in enumerate(loader):
            imgs = imgs.to(config.device)
            
            # Generate captions for the batch
            for i in range(imgs.size(0)):
                img_name = img_names[i]
                
                # Get references (ground truth)
                ref_ids = captions[i].tolist()
                ref_tokens = [vocab.itos[t] for t in ref_ids if t not in [vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]]]
                reference = " ".join(ref_tokens)
                
                if img_name not in gts:
                    gts[img_name] = []
                
                # Avoid duplicates if the loader yields the same caption multiple times (unlikely with standard dataset)
                # But wait, our dataset has 5 entries per image.
                # So we will encounter the same img_name 5 times.
                # We should append the reference each time.
                # Check if this specific reference is already added to avoid duplicates if any
                if not any(d['caption'] == reference for d in gts[img_name]):
                    gts[img_name].append({'caption': reference})
                
                # Only generate if we haven't seen this image before
                if img_name not in res:
                    img = imgs[i]
                    # Use Beam Search for better quality
                    generated_tokens = model.caption_image_beam_search(img, vocab, beam_size=5, alpha=0.7) # Updated beam size
                    if "<SOS>" in generated_tokens: generated_tokens.remove("<SOS>")
                    if "<EOS>" in generated_tokens: generated_tokens.remove("<EOS>")
                    hypothesis = " ".join(generated_tokens)
                    
                    res[img_name] = [{'caption': hypothesis}]

                    # Debug: Print first few examples
                    if len(res) <= 3:
                        print(f"Sample {len(res)} ({img_name}):")
                        print(f"  Ref: {reference}")
                        print(f"  Hyp: {hypothesis}")

    # Convert res to list format expected by scorers (or keep as dict if scorers support it)
    # pycocoevalcap expects res as list of dicts: [{'image_id': id, 'caption': cap}, ...]
    # But wait, the tokenizer expects dict of list of strings if we skip it?
    # Let's stick to the standard format.
    
    res_list = []
    for img_id, caps in res.items():
        res_list.append({'image_id': img_id, 'caption': caps[0]['caption']})
    
    # Tokenization
    print("Tokenizing...")
    if shutil.which('java'):
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize({i['image_id']: [{'caption': i['caption']}] for i in res_list})
    else:
        print("Java not found. Skipping PTBTokenizer and using regex-based tokenizer.")
        # Improved tokenizer to handle punctuation
        import re
        def tokenize_text(text):
            # Split by whitespace and punctuation
            return re.findall(r"\w+|[^\w\s]", text.lower())

        new_gts = {}
        for k, v in gts.items():
            # v is list of dicts: [{'caption': '...'}, ...]
            # We need list of strings, but tokenized
            new_gts[k] = [" ".join(tokenize_text(x['caption'])) for x in v]
            
        new_res = {}
        for item in res_list:
            # item is {'image_id': ..., 'caption': ...}
            new_res[item['image_id']] = [" ".join(tokenize_text(item['caption']))]
            
        gts = new_gts
        res = new_res

    # Calculate Scores
    print("Calculating scores...")
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    
    scores = {}
    for scorer, method in scorers:
        print(f"Computing {method} score...")
        try:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    scores[m] = s
            else:
                scores[method] = score
        except Exception as e:
            print(f"Error computing {method}: {e}")
            if isinstance(method, list):
                for m in method: scores[m] = 0.0
            else:
                scores[method] = 0.0

    return scores

# ==========================================
# 6. Main
# ==========================================
def main():
    # 1. Prepare Data
    prepare_dataset()
    
    # 2. Transforms (Augmentation added)
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])
    
    # 3. Dataset & Loader
    if config.USE_DUMMY_DATA:
        print("Using Dummy Dataset.")
        dataset = DummyDataset(transform=transform_train, freq_threshold=config.freq_threshold)
    elif os.path.exists(config.CAPTION_FILE) and os.path.exists(config.IMG_DIR):
        print("Using Flickr8k Dataset.")
        # Note: We use the same dataset class but different transforms for train/test split
        # Ideally, we should split the dataset first, then apply transforms.
        full_dataset = Flickr8kDataset(config.IMG_DIR, config.CAPTION_FILE, transform=None, freq_threshold=config.freq_threshold)
    else:
        print("Flickr8k Dataset not found and USE_DUMMY_DATA is False. Exiting.")
        return

    # Split Train/Test using Karpathy Split (Standard for Flickr8k)
    # Flickr8k has 8091 images. Standard split is 6000 train, 1000 val, 1000 test.
    # However, the dataset size might vary slightly depending on the version.
    # We will use the first 6000 for train, next 1000 for val, and rest for test.
    
    # Get all unique image names
    all_img_names = sorted(list(set([x[0] for x in full_dataset.captions])))
    random.seed(42) # Ensure reproducibility
    random.shuffle(all_img_names)
    
    train_imgs = set(all_img_names[:6000])
    val_imgs = set(all_img_names[6000:7000])
    test_imgs = set(all_img_names[7000:])
    
    # Create subsets based on image names
    train_indices = [i for i, x in enumerate(full_dataset.captions) if x[0] in train_imgs]
    val_indices = [i for i, x in enumerate(full_dataset.captions) if x[0] in val_imgs]
    test_indices = [i for i, x in enumerate(full_dataset.captions) if x[0] in test_imgs]
    
    train_set = torch.utils.data.Subset(full_dataset, train_indices)
    val_set = torch.utils.data.Subset(full_dataset, val_indices)
    test_set = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Apply transforms manually since we split the dataset
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            img, caption, img_name = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, caption, img_name
            
    # Override __getitem__ in Flickr8kDataset to return PIL image for this to work
    # We need to patch Flickr8kDataset to return PIL image if transform is None
    # (Already done in previous code: if self.transform: image = self.transform(image))
    
    train_set = TransformedSubset(train_set, transform_train)
    val_set = TransformedSubset(val_set, transform_test) # Use test transform for val
    test_set = TransformedSubset(test_set, transform_test)
    
    pad_idx = full_dataset.vocab.stoi["<PAD>"]
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=Collate(pad_idx), num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, collate_fn=Collate(pad_idx), num_workers=4)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=Collate(pad_idx), num_workers=4)
    
    vocab_size = len(full_dataset.vocab)
    vocab = full_dataset.vocab

    # 4. Model
    print(f"Initializing Model with Vocab Size: {vocab_size}")
    model = ImageCaptionModel(
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        vocab_size=vocab_size,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_idx=pad_idx,
        train_CNN=True # Enable fine-tuning
    ).to(config.device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"], label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # 5. Training Loop
    history = {
        'loss': [], 'Bleu_1': [], 'Bleu_4': [], 
        'ROUGE_L': [], 'CIDEr': []
    }
    
    best_score = 0.0

    for epoch in range(config.num_epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, epoch)
        history['loss'].append(loss)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}")
        
        scheduler.step(loss)
        
        # Evaluate every epoch (or every few epochs to save time)
        print("Evaluating on Validation Set...")
        # Use val_loader for model selection
        scores = evaluate_metrics(model, val_loader, vocab)
        for k, v in scores.items():
            if k in history:
                history[k].append(v)
            print(f"{k}: {v:.4f}")
            
        # Save best model based on CIDEr score
        current_score = scores.get('CIDEr', 0.0)
        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
            print(f"New best model saved with CIDEr: {best_score:.4f}")
            
    # Final Evaluation on Test Set
    print("Loading best model for final testing...")
    model.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "best_model.pth")))
    print("Evaluating on Test Set...")
    test_scores = evaluate_metrics(model, test_loader, vocab)
    print("Final Test Scores:")
    for k, v in test_scores.items():
        print(f"{k}: {v:.4f}")

    # 7. Plotting (6 separate plots)
    metrics_to_plot = ['loss', 'Bleu_1', 'Bleu_4', 'ROUGE_L', 'CIDEr']
    
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in history and len(history[metric]) > 0:
            axes[i].plot(range(1, len(history[metric]) + 1), history[metric], marker='o')
            title_prefix = "Training" if metric == "loss" else "Validation"
            axes[i].set_title(f'{title_prefix} {metric}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric)
            axes[i].grid(True)
        else:
            axes[i].text(0.5, 0.5, 'No Data', ha='center')
            
    # Hide unused subplots
    for j in range(len(metrics_to_plot), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('experiment8_metrics_summary.png')
    print("Metrics summary plot saved to experiment8_metrics_summary.png")

if __name__ == "__main__":
    main()
