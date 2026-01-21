import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

'''
数据准备
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
def prepare_data():
    datas = np.load("tang.npz", allow_pickle=True)
    data = torch.from_numpy(datas['data'])
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()

    dataloader = DataLoader(data, batch_size=32, shuffle=True, num_workers=2,pin_memory=(device.type == "cuda"))
    PAD = word2ix['<PAD>']
    UNK = word2ix['<UNK>']
    return dataloader, ix2word, word2ix, PAD, UNK


'''
模型设计，采用多层LSTM，增加dropout
'''
class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,num_layers=2, padding_idx=0):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim,padding_idx=padding_idx)   #将字符索引编码成定长向量
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=0.3, batch_first=True)
        #输入某一个字符串的向量表示，对每个字对应的向量输出一个output：隐藏层维数的向量，h_n：字符串最后一个字的Hidden State，c_n：最后一个字的Cell State。
        self.linear = nn.Linear(hidden_dim, vocab_size)#全连接层，将隐藏状态（dim=hidden_dim ）转化为字的出现概率

    def forward(self, input, hidden=None):
        embeds = self.embeddings(input)
        if hidden is None:
            batch_size = input.size(0)
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(input.device)  #在第一次传播定义0向量。2是因为，有两层LSTM，返回h与c在这样定义的直积空间下维数为2
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(input.device)
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output)
        output = output.reshape(-1, output.size(2))
        return output, hidden
'''
模型训练
'''

'''
湖光秋月两相和
人生若只如初见
世事一场大梦
云想衣裳花想容
'''
# def train(model, dataloader,PAD_IDX,ix2word,word2ix,UNK_IDX,epochs=10, lr=3e-4, generate_every=500,start_words='云想衣裳花想容'
#           ,max_gen_len=50):
#     optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)
#     total_steps = len(dataloader) * epochs
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
#                                                            T_max=total_steps)
#     criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
#     model.train()


#     fig, ax = plt.subplots()
#     plot_every = 100
#     batch_losses, steps = [], []
#     global_step = 0
#     for epoch in range(epochs):
#         total_loss = 0
#         for data in dataloader:
#             data = data.long() #转化成整数张量
#             input_data = data[:, :-1].to(device)  #去掉最后一个词作为输入
#             target = data[:, 1:].contiguous().view(-1).to(device) #去掉第一个词作为目标，压平成一维，计算交叉熵用
#             optimizer.zero_grad()
#             output, _ = model(input_data)
#             loss = criterion(output, target)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#             optimizer.step()
#             scheduler.step()

#             global_step += 1
#             batch_losses.append(loss.item())
#             steps.append(global_step)
#             total_loss += loss.item()

#             if global_step % generate_every == 0:
#                 model.eval()                        # 关闭 dropout 等
#                 with torch.no_grad():
#                     poem = generate(model,
#                                     start_words=start_words,
#                                     ix2word=ix2word,
#                                     word2ix=word2ix,
#                                     PAD_IDX=PAD_IDX,
#                                     UNK_IDX=UNK_IDX,
#                                     max_len=max_gen_len)
#                 print(f"\n[Step {global_step}] 生成诗句：{poem}\n")
#                 model.train()
#             # ---- 只在指定步数刷新一次 ----
#             if global_step % plot_every == 0:
#                 ax.cla()  # ax.clear() 亦可
#                 ax.plot(steps, batch_losses)
#                 ax.set_xlabel('Batch #')
#                 ax.set_ylabel('Loss')
#                 plt.pause(0.01)
#         avg_loss = total_loss / len(dataloader)
#         print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")
#     plt.savefig('3_train_accuracy_curve.png', dpi=300)
#     plt.show()

def train(model, dataloader,PAD_IDX,ix2word,word2ix,UNK_IDX,epochs=10, lr=3e-4, generate_every=500,start_words='云想衣裳花想容'
          ,max_gen_len=50):
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)
    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    model.train()


    fig, ax = plt.subplots()
    plot_every = 100
    batch_losses, steps = [], []
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            data = data.long() #转化成整数张量
            input_data = data[:, :-1].to(device)  #去掉最后一个词作为输入
            target = data[:, 1:].contiguous().view(-1).to(device) #去掉第一个词作为目标，压平成一维，计算交叉熵用
            optimizer.zero_grad()
            output, _ = model(input_data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            batch_losses.append(loss.item())
            steps.append(global_step)
            total_loss += loss.item()

            if global_step % generate_every == 0:
                model.eval()                        # 关闭 dropout 等
                with torch.no_grad():
                    # 在训练过程中使用不同的温度参数来展示效果
                    temperatures = [0.5, 0.8, 1.0]
                    for temp in temperatures:
                        poem = generate(model,
                                      start_words=start_words,
                                      ix2word=ix2word,
                                      word2ix=word2ix,
                                      PAD_IDX=PAD_IDX,
                                      UNK_IDX=UNK_IDX,
                                      max_len=max_gen_len,
                                      temperature=temp,  # 添加温度参数
                                      top_k=15)          # 添加top-k参数
                        print(f"[Step {global_step}, temp={temp}] 生成诗句：{poem}")
                print()  # 空行分隔
                model.train()
            # ---- 只在指定步数刷新一次 ----
            if global_step % plot_every == 0:
                ax.cla()  # ax.clear() 亦可
                ax.plot(steps, batch_losses)
                ax.set_xlabel('Batch #')
                ax.set_ylabel('Loss')
                plt.pause(0.01)
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")
    plt.savefig('3_train_accuracy_curve.png', dpi=300)
    plt.show()

'''
预测
'''

# def generate(model, start_words, ix2word, word2ix, PAD_IDX, UNK_IDX, max_len=50, temperature=0.8, top_k=10):
#     results = list(start_words)
#     model.eval()
#     input = torch.tensor([[word2ix['<START>']]],device=device).long()
#     hidden = None
#     with torch.no_grad():
#         for i in range(max_len):
#             # 输入给定的句首
#             output, hidden = model(input, hidden)
#             if i < len(start_words):
#                 w = results[i]
#                 input = torch.tensor([[word2ix.get(w, word2ix['<UNK>'])]],device=device).long()
#             else:
#                 top_index = output[-1].topk(1)[1][0].item()
#                 w = ix2word[top_index]
#                 if w == '<EOP>':
#                     break
#                 results.append(w)
#                 input = torch.tensor([[top_index]],device=device).long()
#     return ''.join(results)

def generate(model, start_words, ix2word, word2ix, PAD_IDX, UNK_IDX, max_len=50, temperature=0.8, top_k=10):
    """
    改进的生成函数，加入温度采样和top-k采样
    temperature: 控制随机性，值越小越确定，值越大越随机
    top_k: 只从概率最高的k个词中采样
    """
    results = list(start_words)
    model.eval()
    input = torch.tensor([[word2ix['<START>']]], device=device).long()
    hidden = None
    
    with torch.no_grad():
        for i in range(max_len):
            output, hidden = model(input, hidden)
            
            if i < len(start_words):
                # 输入给定的句首
                w = results[i]
                input = torch.tensor([[word2ix.get(w, word2ix['<UNK>'])]], device=device).long()
            else:
                # 改进的采样策略
                logits = output[-1]  # 取最后一个时间步的输出
                
                # 应用温度
                logits = logits / temperature
                
                # top-k 过滤
                if top_k > 0:
                    # 获取top-k的值和索引
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    # 创建与logits相同形状的mask，初始为负无穷
                    indices_to_remove = logits < top_k_logits[-1]
                    logits[indices_to_remove] = -float('Inf')
                
                # softmax得到概率分布
                probabilities = torch.softmax(logits, dim=-1)
                
                # 从概率分布中采样
                top_index = torch.multinomial(probabilities, 1).item()
                
                w = ix2word[top_index]
                if w == '<EOP>':
                    break
                results.append(w)
                input = torch.tensor([[top_index]], device=device).long()
    
    return ''.join(results)

class Config:
    embedding_dim = 128
    hidden_dim = 384
    batch_size = 32
    lr = 5e-4
    epochs = 10
    max_gen_len = 50


# def main():
#     dataloader, ix2word, word2ix, PAD_IDX, UNK_IDX = prepare_data()
#     print("==> 准备数据1")

#     print("==> 初始化模型")
#     model = PoetryModel(vocab_size=len(word2ix),
#         embedding_dim=Config.embedding_dim,
#         hidden_dim=Config.hidden_dim,
#         num_layers=2,
#         padding_idx=PAD_IDX).to(device)

#     print("==> 开始训练")
#     train(model, dataloader,PAD_IDX,ix2word, word2ix, UNK_IDX,epochs=Config.epochs, lr=Config.lr)

#     poem = generate(model,
#                     start_words='湖光秋月两相和',
#                     ix2word=ix2word,
#                     word2ix=word2ix,
#                     PAD_IDX=PAD_IDX,
#                     UNK_IDX=UNK_IDX,
#                     max_len=Config.max_gen_len)
#     print("生成诗句：", poem)

def main():
    dataloader, ix2word, word2ix, PAD_IDX, UNK_IDX = prepare_data()
    print("==> 准备数据完成")

    print("==> 初始化模型")
    model = PoetryModel(vocab_size=len(word2ix),
        embedding_dim=Config.embedding_dim,
        hidden_dim=Config.hidden_dim,
        num_layers=2,
        padding_idx=PAD_IDX).to(device)

    print("==> 开始训练")
    train(model, dataloader, PAD_IDX, ix2word, word2ix, UNK_IDX, epochs=Config.epochs, lr=Config.lr)

    # 训练完成后，尝试不同的参数组合来生成诗句
    test_sentences = ['湖光秋月两相和', '人生若只如初见', '春风又绿江南岸']
    
    for start_sentence in test_sentences:
        print(f"\n=== 起始句: {start_sentence} ===")
        
        # 尝试不同的温度设置
        for temperature in [0.6, 0.8, 1.0]:
            poem = generate(model,
                          start_words=start_sentence,
                          ix2word=ix2word,
                          word2ix=word2ix,
                          PAD_IDX=PAD_IDX,
                          UNK_IDX=UNK_IDX,
                          max_len=Config.max_gen_len,
                          temperature=temperature,
                          top_k=12)
            print(f"温度={temperature}: {poem}")


if __name__ == "__main__":
    main()