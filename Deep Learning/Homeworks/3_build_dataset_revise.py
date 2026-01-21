import os
import json
import numpy as np
import re
import argparse
from pathlib import Path

TOKEN_RE = re.compile(r'<START>|<EOP>|.', re.S)

# 1. 读取文件夹中所有诗词
"""
    读取 chinese-poetry 项目中的 poet.tang.*.json / poet.song.*.json 等文件，
    把每首诗的 paragraphs 拼成一个字符串，过滤长度后返回列表。
    json格式（自带标点）；
    [
  {
    "author": "太宗皇帝",
    "paragraphs": [
      "秦川雄帝宅，函谷壯皇居。",
      "綺殿千尋起，離宮百雉餘。",
      "連甍遙接漢，飛觀迥凌虛。",
      "雲日隱層闕，風煙出綺疎。"
    ],
    "note": [],
    "title": "帝京篇十首 一"
  }
]
"""
def read_poems_from_json_folder(folder_path, max_len=125):

    poems = []

    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'):
            continue

        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)  #data是一个列表，每个列表是对每一首诗的字典
            for poem_dict in data:
                if not isinstance(poem_dict, dict):
                    continue  # 跳过不是字典的元素
                para_list = poem_dict.get('paragraphs', []) #para_list 是一个列表，元素是诗的每一段
                text = ''.join(para_list)
                text = text.replace(' ', '').replace('\u3000', '')

                # 过滤极短或极长的样本
                if not (5 <= len(text) <= max_len):
                    continue
                poems.append('<START>' + text + '<EOP>')

    return poems
def tokenize(poem):   # 将诗歌切分成一个个token，但不切分start和eop
    return TOKEN_RE.findall(poem)
# 2. 构建词表
def build_vocab(poems):
    # 加入 4 个特殊符号
    specials = ['<PAD>', '<START>', '<EOP>', '<UNK>']
    word2ix = {w: i for i, w in enumerate(specials)} #先将特殊字符单独塞进word2ix

    tokens = set()
    for poem in poems:
        tokens.update(tokenize(poem))

    tokens -= set(specials)            # 避免重复添加特殊标记
    tokens = sorted(tokens)            # 词表顺序固定，保证复现

    for t in tokens:
        word2ix[t] = len(word2ix)  #t是poem中不含特殊字符的字符，如“秦”，将其逐个按顺序排到word2ix的最后分配索引

    ix2word = {i: w for w, i in word2ix.items()}
    return word2ix, ix2word
# 3. 转换为索引并填充
def poems_to_tensor(poems, word2ix, max_len=125):
    data = []
    PAD = word2ix['<PAD>']
    UNK = word2ix['<UNK>']  #提取出PAD与UNK的索引

    for poem in poems:
        tokens = tokenize(poem)  #将poem拆成token，但<start>保留
        poem_ix = [word2ix.get(w, UNK) for w in tokens]  #从tokens中的每个字符逐个映射到word2ix中的索引（字 → 索引），如果某个字没找到，用UNK的字符串替代。
        if len(poem_ix) < max_len:
            poem_ix += [PAD]* (max_len - len(poem_ix))  #如果小于最大字符长度，则用<PAD>的索引填充
        else:
            poem_ix = poem_ix[:max_len]
        data.append(poem_ix)
        #data中储存的是每一首诗的索引
    return np.array(data)

# 4. 主函数
def find_poems_folder(explicit_path: str = None):
    """
    找到一个合适的诗歌 json 文件夹。

    优先权：
    1. explicit_path（如果提供且存在）
    2. 尝试此脚本旁边的几个候选相对位置
    3. 在几个父级中搜索一个名为“chinese-poetry-master”的文件夹，其中包含“全唐诗”
    返回第一个现有文件夹路径 （str） 或 None （如果未找到）。
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return str(p)

    script_dir = Path(__file__).resolve().parent

    # common candidate relative locations
    candidates = [
        script_dir / 'chinese-poetry-master' / '全唐诗',
        script_dir.parent / 'chinese-poetry-master' / '全唐诗',
        script_dir.parent.parent / 'chinese-poetry-master' / '全唐诗',
        script_dir / '..' / '作业-学长' / '作业三' / 'chinese-poetry-master' / '全唐诗',
    ]

    for c in candidates:
        c = c.resolve()
        if c.exists():
            return str(c)

    # search upward a few levels and walk directories to locate 'chinese-poetry-master/全唐诗'
    for up in range(0, 4):
        root = (script_dir / ('..' * up)).resolve()
        # guard: only search if root is inside a reasonable path
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                # look for folder named chinese-poetry-master
                base = Path(dirpath)
                if base.name == 'chinese-poetry-master':
                    cand = base / '全唐诗'
                    if cand.exists():
                        return str(cand)
                # small optimization: if dirpath already deep, skip deep traversals
                # limit walking depth relative to root
                if len(Path(dirpath).relative_to(root).parts) > 4:
                    # don't traverse deeper under this root
                    dirnames[:] = []
        except Exception:
            continue

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build dataset from chinese-poetry json files')
    parser.add_argument('--folder', type=str, default=None,
                        help='path to the folder containing json files, e.g. /path/to/chinese-poetry-master/全唐诗')
    args = parser.parse_args()

    folder = find_poems_folder(args.folder)
    if folder is None:
        print('\nERROR: 无法找到诗词数据文件夹。请确认你已将 chinese-poetry-master 下载到工作区，或使用 --folder 指定路径。')
        print('常见修复：')
        print('  1) 把项目目录放在与脚本相同的上层目录，或')
        print('  2) 使用绝对路径运行: python 3_build_dataset_revise.py --folder D:/path/to/chinese-poetry-master/全唐诗')
        print('\n脚本当前尝试的位置示例（按优先级搜索）:')
        print(f'  当前脚本目录: {Path(__file__).resolve().parent}')
        raise SystemExit(2)

    poems = read_poems_from_json_folder(folder)
    word2ix, ix2word = build_vocab(poems)
    data = poems_to_tensor(poems, word2ix)

    np.savez('tang.npz', data=data, word2ix=word2ix, ix2word=ix2word)
    print(f"保存完成，共{len(poems)}首诗，词表大小：{len(word2ix)}")