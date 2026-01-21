import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import json
import pathlib
import tarfile
import shutil
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import sacrebleu
import matplotlib


# ===== 数据预处理 =====
def preprocess_data(extract_dir: str = "data"):
    """准备并切分平行语料，必要时自动查找或解压数据文件。"""

    def locate_data_files():
        script_dir = pathlib.Path(__file__).resolve().parent
        candidates = [
            pathlib.Path('.').resolve(),
            script_dir,
            script_dir / 'data',
            script_dir.parent,
            script_dir.parent / 'data'
        ]

        for base in candidates:
            zh = base / 'chinese.txt'
            en = base / 'english.txt'
            if zh.exists() and en.exists():
                return zh, en

        # limited depth search inside candidate dirs
        for base in candidates:
            if base.exists():
                for child in base.iterdir():
                    if not child.is_dir():
                        continue
                    zh = child / 'chinese.txt'
                    en = child / 'english.txt'
                    if zh.exists() and en.exists():
                        return zh, en

        # fallback deep search (bounded) in script dir
        count = 0
        for zh in script_dir.rglob('chinese.txt'):
            if count >= 300:
                break
            en = zh.parent / 'english.txt'
            if en.exists():
                return zh, en
            count += 1

        return None, None

    def try_extract_from_tar(dest: str):
        for candidate in pathlib.Path('.').rglob('*.tar.gz'):
            try:
                with tarfile.open(candidate, 'r:gz') as tar:
                    tar.extractall(path=dest)
                return True
            except Exception:
                continue
        return False

    zh_path, en_path = locate_data_files()
    if zh_path is None or en_path is None:
        pathlib.Path(extract_dir).mkdir(parents=True, exist_ok=True)
        if try_extract_from_tar(extract_dir):
            zh_path, en_path = locate_data_files()

    if zh_path is None or en_path is None:
        print("错误：找不到 chinese.txt / english.txt。请确保数据解压在项目目录或 data/ 中。")
        print("当前目录列表（最多30项）：")
        for i, item in enumerate(pathlib.Path('.').iterdir()):
            if i >= 30:
                break
            print('  -', item)
        raise FileNotFoundError("chinese.txt / english.txt not found")

    zh_lines = zh_path.read_text(encoding='utf8').strip().splitlines()
    en_lines = en_path.read_text(encoding='utf8').strip().splitlines()
    assert len(zh_lines) == len(en_lines)
    # 读取文件，strip()去除首尾空白符，splitlines()按换行符将文件拆分成list，assert确保中英文一一对应

    data = list(zip(zh_lines, en_lines))
    random.shuffle(data)
    # data是一个list，元素是(中文句子, 英文句子)的tuple，zh_lines和en_lines中的元素（每一行句子）组成tuple

    n = len(data)
    train, dev, test = data[:int(0.8*n)], data[int(0.8*n):int(0.9*n)], data[int(0.9*n):]  # 按照8:1:1划分训练集、验证集、测试集

    for split, name in [(train, "train"), (dev, "dev"), (test, "test")]:  # 循环三次，第一次处理train，第二次dev，第三次test；split是对应的数据集，name是对应的文件名前缀
        with open(f"{name}.zh", "w", encoding="utf8") as fz, \
             open(f"{name}.en", "w", encoding="utf8") as fe:
            for zh, en in split:
                fz.write(zh.strip() + "\n")
                fe.write(en.strip() + "\n")
    # 将划分好的数据集分别写入train.zh/train.en, dev.zh/dev.en, test.zh/test.en文件中

# 新建词表文件，取数据集中的高频词，并且输出词表
def build_vocab(path, max_size=30000):
    counter = Counter()
    specials = ["<pad>", "<unk>", "<bos>", "<eos>"]  # 特殊符号，分别表示填充、未知词、句子开始、句子结束
    
    for line in pathlib.Path(path).read_text(encoding="utf8").splitlines():
        counter.update(line.split())  # line是读取文件中的str（如读取train.zh，其中是一个列表，元素是一句话。split按空格分词，每一次循环counter更新词的数目。）
    
    most_common = [w for w, _ in counter.most_common(max_size - len(specials))]  # 取counter中max_size - len(specials)个词作为一个list
    words = specials + most_common
    stoi = {w: i for i, w in enumerate(words)}
    itos = {i: w for w, i in stoi.items()}
    
    with open(path + ".vocab.json", "w", encoding="utf8") as f:
        json.dump(stoi, f)  # 将词表以json格式保存到文件中，文件名为原文件名加.vocab.json后缀
    
    return stoi, itos

# zh_stoi, zh_itos = build_vocab("train.zh")
# en_stoi, en_itos = build_vocab("train.en")

# ===== 数据集类 =====
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, zh_path, en_path, zh_vocab, en_vocab, max_len=100):
        self.zh = pathlib.Path(zh_path).read_text(encoding='utf8').splitlines()
        self.en = pathlib.Path(en_path).read_text(encoding='utf8').splitlines()
        self.zstoi, self.estoi = zh_vocab, en_vocab
        self.max_len = max_len
    
    def encode(self, sent, vocab, bos_eos=True):  # sent为输入的一句话，将其转换为token
        ids = [vocab.get(tok, vocab['<unk>']) for tok in sent.split()][:self.max_len-2]
        # 建立一个str->token的映射表，split()按空格分词，get(tok, vocab['<unk>'])表示如果词在词表中找不到，则用<unk>代替
        if bos_eos:
            return [vocab["<bos>"]] + ids + [vocab["<eos>"]]
        return ids
    
    def pad(self, ids, pad_id):
        return ids + [pad_id] * (self.max_len - len(ids))
    # 补位，sent过短情况下len(ids) < max_len，则在ids后面补pad_id直到长度为max_len
    
    def __len__(self):
        return len(self.zh)  # 返回数据集的大小，即句子对的数量
    
    def __getitem__(self, idx):
        src = self.encode(self.zh[idx], self.zstoi)
        tgt = self.encode(self.en[idx], self.estoi)
        return (
            torch.tensor(self.pad(src, self.zstoi["<pad>"])),
            torch.tensor(self.pad(tgt, self.estoi["<pad>"]))
        )
    # 将中文编码与英文编码token组成一个tuple，输入dataset

# train_ds = TranslationDataset("train.zh", "train.en", zh_stoi, en_stoi)
# dev_ds = TranslationDataset("dev.zh", "dev.en", zh_stoi, en_stoi)
# train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

# ===== 模型组件 =====
def clones(module, N):
    return nn.ModuleList([module if i == 0 else type(module)(*module.init_args) for i in range(N)])

class MultiHeadAttention(nn.Module):  # 多头注意力机制
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0  # 确保输入维数平均分给每个头
        self.d_model, self.n_head = d_model, n_head
        self.d_k = d_model // n_head  # 每个头的维度
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.init_args = (d_model, n_head, dropout)  # 前面拷贝中用，转化为tuple
    
    def forward(self, q, k, v, mask=None):
        B, L_q, _ = q.size()
        B, L_k, _ = k.size()
        
        q = self.w_q(q).view(B, L_q, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, L_k, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, L_k, self.n_head, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask是一个布尔矩阵，将其为0的位置对应的scores设为一个很小的数，防止softmax后被选中
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        return self.w_o(context)

class PositionwiseFF(nn.Module):  # 前馈网络
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.init_args = (d_model, d_ff, dropout)
    
    def forward(self, x):
        return self.net(x)

class PositionalEncoding(nn.Module):  # 位置编码
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)  # 形状(max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)  # 偶数位置
        pe[:, 1::2] = torch.cos(pos * div)  # 奇数位置
        self.register_buffer('pe', pe.unsqueeze(0))  # 形状(1, max_len, d_model//2)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  #x.size(0)=batch, x.size(1)=seq_len x_size(2)=d_model

class EncoderLayer(nn.Module):  #Encoder
    def __init__(self, d_model, n_head, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ff = PositionwiseFF(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.init_args = (d_model, n_head, d_ff, dropout)
    
    def forward(self, x, src_mask):
        # 自注意力子层
        attn = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn))
        # 前馈网络子层
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        return x

class Encoder(nn.Module):  # 整体堆叠
    def __init__(self, vocab_size, d_model, M=6, n_head=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)  # 将词表映射成d_model维向量
        self.pos = PositionalEncoding(d_model)
        layer = EncoderLayer(d_model, n_head, d_ff, dropout)
        self.layers = clones(layer, M)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src, src_mask):
        x = self.pos(self.embed(src) * math.sqrt(self.embed.embedding_dim))  # 对输入的整数ID向量进行词嵌入，并加上位置向量 维数（batch, len, d_model）
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):  # Decoder
    def __init__(self, d_model, n_head, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ff = PositionwiseFF(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.init_args = (d_model, n_head, d_ff, dropout)
    
    def forward(self, x, memory, tgt_mask, src_mask):
        # 自注意力子层：seft-attention + 残差连接 + 层归一化
        attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn))
        # 交叉注意力子层：cross-attention + 残差连接 + 层归一化
        attn2 = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        # 前馈网络子层：FFN + 残差连接 + 层归一化
        ff = self.ff(x)
        x = self.norm3(x + self.dropout(ff))
        return x

class Decoder(nn.Module):  # 整体堆叠
    def __init__(self, vocab_size, d_model, N=6, n_head=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = DecoderLayer(d_model, n_head, d_ff, dropout)
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, tgt, memory, tgt_mask, src_mask):
        x = self.pos(self.embed(tgt) * math.sqrt(self.embed.embedding_dim))
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)

class TransformerNMT(nn.Module):  # 整体模型
    def __init__(self, src_vocab, tgt_vocab, d_model=512, M=6, n_head=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(len(src_vocab), d_model, M, n_head, d_ff, dropout)
        self.decoder = Decoder(len(tgt_vocab), d_model, M, n_head, d_ff, dropout)
        self.generator = nn.Linear(d_model, len(tgt_vocab))  # 将decoder的输出向量映射到词表上的logits，每个位置对应一个词的概率分布
        self.src_pad = src_vocab['<pad>']
        self.tgt_pad = tgt_vocab['<pad>']
        self.tgt_bos = tgt_vocab['<bos>']
        self.tgt_eos = tgt_vocab['<eos>']
    
    def make_masks(self, src, tgt):
        src_mask = (src != self.src_pad).unsqueeze(1).unsqueeze(2)
        tgt_pad_mask = (tgt != self.tgt_pad).unsqueeze(1).unsqueeze(2)
        size = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones(size, size, device=src.device)).bool()
        # tgt_mask = tgt_pad_mask & (~tgt_sub_mask)
        tgt_mask = tgt_pad_mask & tgt_sub_mask  # 屏蔽<pad>和未来信息，注意这里是取反，是一个包含对角线的下三角矩阵，与多头注意力点积一致
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.make_masks(src, tgt[:, :-1])
        memory = self.encoder(src, src_mask)
        out = self.decoder(tgt[:, :-1], memory, tgt_mask, src_mask)
        logits = self.generator(out)
        return logits

# ===== 推理和评估 =====
def translate_batch(model, src_batch, max_len=80, device="cpu"):
    model.eval()
    src_batch = src_batch.to(device)
    
    src_mask = (src_batch != model.src_pad).unsqueeze(1).unsqueeze(2)
    
    with torch.no_grad():
        memory = model.encoder(src_batch, src_mask)
    
    ys = torch.full((src_batch.size(0), 1), model.tgt_bos, dtype=torch.long, device=device)
    
    for _ in range(max_len):
        _, tgt_mask = model.make_masks(src_batch, ys)
        out = model.decoder(ys, memory, tgt_mask, src_mask)
        logits = model.generator(out[:, -1])  # 取最后一个时间步的输出，生成字符b.len(vocab)
        next_tok = logits.argmax(-1, keepdim=True)  # 选取词表中概率最大的一个词。-1表示最后一个维度，即词表维度
        ys = torch.cat([ys, next_tok], dim=1)  # b,2 (or b, k+1)
        
        if (next_tok == model.tgt_eos).all():  # 若batch中所有句子都生成了<eos>，则停止生成
            break
    
    return ys

def ids_to_sent(ids, itos, eos_id, pad_id):  # 把一串id转成字符串，遇到<eos>截断，跳过<pad>
    words = []
    for idx in ids:
        if idx == eos_id:
            break
        if idx != pad_id:
            words.append(itos[idx])
    return " ".join(words)

def evaluate_bleu(model, dataset, en_itos, batch_size=64, device="cpu"):
    model.eval()
    sys, refs = [], []
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for src, tgt in dl:
            ys = translate_batch(model, src, device=device)
            ys = ys.cpu().tolist()
            tgt = tgt.tolist()
            
            for hyp_ids, ref_ids in zip(ys, tgt):
                hyp = ids_to_sent(hyp_ids[1:], en_itos, model.tgt_eos, model.tgt_pad)  # 去掉开头的<BOS>
                ref = ids_to_sent(ref_ids[1:], en_itos, model.tgt_eos, model.tgt_pad)
                sys.append(hyp)
                refs.append([ref])
    
    bleu = sacrebleu.corpus_bleu(sys, refs).score
    return bleu

# ===== 训练代码 =====
class Meter:
    def __init__(self, bleu_interval=2500):
        self.bleu_interval = bleu_interval
        self.step = 0
        self.history = {'loss': [], 'bleu': []}
    
    def update_loss(self, loss):
        self.step += 1
        self.history['loss'].append((self.step, loss))
    
    def maybe_eval_bleu(self, model, dev_ds, en_itos, device):
        if self.step % self.bleu_interval == 0:
            bleu = evaluate_bleu(model, dev_ds, en_itos, device=device)
            self.history['bleu'].append((self.step, bleu))
            print(f"[step {self.step}] BLEU = {bleu:.2f}")
    
    def plot(self):
        """绘制loss和BLEU曲线图"""
        try:
            # 使用Agg后端，不显示图形窗口
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # 创建两个子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # 绘制loss曲线
            if self.history['loss']:
                steps, losses = zip(*self.history['loss'])
                ax1.plot(steps, losses, 'b-', alpha=0.7, linewidth=1)
                ax1.set_xlabel('Training Step')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training Loss')
                ax1.grid(True, alpha=0.3)
                
                # 添加趋势线（使用滑动平均）
                if len(losses) > 10:
                    window_size = max(10, len(losses) // 20)
                    trend = []
                    for i in range(len(losses)):
                        start = max(0, i - window_size // 2)
                        end = min(len(losses), i + window_size // 2 + 1)
                        trend.append(sum(losses[start:end]) / (end - start))
                    ax1.plot(steps, trend, 'r-', linewidth=2, alpha=0.8, label='Trend')
                    ax1.legend()
            
            # 绘制BLEU曲线
            if self.history['bleu']:
                steps, bleus = zip(*self.history['bleu'])
                ax2.plot(steps, bleus, 'g-', marker='o', markersize=4, linewidth=2)
                ax2.set_xlabel('Training Step')
                ax2.set_ylabel('BLEU Score')
                ax2.set_title('BLEU Score on Dev Set')
                ax2.grid(True, alpha=0.3)
                
                # 在点上标注数值
                for step, bleu in zip(steps, bleus):
                    ax2.annotate(f'{bleu:.1f}', (step, bleu), 
                               textcoords="offset points", xytext=(0,8), 
                               ha='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
            print("训练曲线已保存为 'training_curves.png'")
            
            # 分别保存loss和BLEU图
            if self.history['loss']:
                plt.figure(figsize=(6, 4))
                steps, losses = zip(*self.history['loss'])
                plt.plot(steps, losses, 'b-', alpha=0.7, linewidth=1)
                plt.xlabel('Training Step')
                plt.ylabel('Loss')
                plt.title('Training Loss')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
                print("损失曲线已保存为 'loss_curve.png'")
            
            if self.history['bleu']:
                plt.figure(figsize=(6, 4))
                steps, bleus = zip(*self.history['bleu'])
                plt.plot(steps, bleus, 'g-', marker='o', markersize=4, linewidth=2)
                plt.xlabel('Training Step')
                plt.ylabel('BLEU Score')
                plt.title('BLEU Score on Dev Set')
                plt.grid(True, alpha=0.3)
                
                # 在点上标注数值
                for step, bleu in zip(steps, bleus):
                    plt.annotate(f'{bleu:.1f}', (step, bleu), 
                               textcoords="offset points", xytext=(0,8), 
                               ha='center', fontsize=8)
                
                plt.tight_layout()
                plt.savefig('bleu_curve.png', dpi=300, bbox_inches='tight')
                print("BLEU曲线已保存为 'bleu_curve.png'")
                
        except ImportError:
            print("警告: 无法导入matplotlib.pyplot，无法绘制训练曲线")
        except Exception as e:
            print(f"绘图时出错: {e}")

# 主训练循环
if __name__ == "__main__":
    # 数据预处理
    preprocess_data()
    
    # 构建词表
    zh_stoi, zh_itos = build_vocab("train.zh")
    en_stoi, en_itos = build_vocab("train.en")
    
    # 创建数据集
    train_ds = TranslationDataset("train.zh", "train.en", zh_stoi, en_stoi)
    dev_ds = TranslationDataset("dev.zh", "dev.en", zh_stoi, en_stoi)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模型初始化
    model = TransformerNMT(zh_stoi, en_stoi).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    criterion = nn.CrossEntropyLoss(ignore_index=en_stoi["<pad>"])
    meter = Meter(bleu_interval=2500)  # 每2500步评估一次BLEU
    
    # 训练函数
    def run_epoch(dataloader, train=True):
        model.train(train)
        total_loss, n_tok = 0, 0
        
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            n_tok += (tgt[:, 1:] != en_stoi["<pad>"]).sum().item()
            total_loss += loss.item() * tgt[:, 1:].numel()
            meter.update_loss(loss.item())
            meter.maybe_eval_bleu(model, dev_ds, en_itos, device)
            
            if train and meter.step % 10 == 0:
                print(f"[step {meter.step}] loss={loss.item():.3f}")
        
        return total_loss / n_tok
    
    # 训练循环
    best_bleu = 0
    for epoch in range(6):
        train_loss = run_epoch(train_dl, train=True)
        
        # 在每个epoch结束时也计算一次BLEU
        if meter.history['bleu']:
            dev_bleu = meter.history['bleu'][-1][1]
        else:
            # 如果还没有计算过BLEU，现在计算一次
            dev_bleu = evaluate_bleu(model, dev_ds, en_itos, device=device)
            meter.history['bleu'].append((meter.step, dev_bleu))
            print(f"[epoch {epoch} end] BLEU = {dev_bleu:.2f}")
        
        if dev_bleu > best_bleu:
            torch.save(model.state_dict(), "best.pt")
            best_bleu = dev_bleu
        
        print(f"Epoch {epoch:02d} | Train loss {train_loss:.3f} | Dev BLEU {dev_bleu:.2f}")
    
    # 训练结束后绘制曲线
    meter.plot()