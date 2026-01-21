import os
import requests
import zipfile
import tarfile
import pandas as pd
from pathlib import Path
import shutil

class NiuTransDataset:
    def __init__(self, data_dir="niu_trans_data"):
        self.data_dir = Path(data_dir)
        self.base_url = "https://raw.githubusercontent.com/NiuTrans/NiuTrans.SMT/master/sample-data"
        
    def download_file(self, url, filename):
        """下载单个文件"""
        local_path = self.data_dir / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"正在下载: {filename}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"下载完成: {filename}")
            return True
        except Exception as e:
            print(f"下载失败 {filename}: {e}")
            return False
    
    def download_dataset(self):
        """下载整个数据集"""
        print("开始下载NiuTrans数据集...")
        
        # 文件列表
        files = [
            # TM-training-set
            "TM-training-set/chinese.txt",
            "TM-training-set/english.txt", 
            "TM-training-set/chinese.tree.txt",
            "TM-training-set/english.tree.txt",
            "TM-training-set/Alignment.txt",
            
            # Dev-set
            "Dev-set/Niu.dev.txt",
            
            # Test-set
            "Test-set/Niu.test.txt",
            
            # reference-set
            "reference-set/Niu.test.reference"
        ]
        
        success_count = 0
        for file_path in files:
            url = f"{self.base_url}/{file_path}"
            if self.download_file(url, file_path):
                success_count += 1
        
        print(f"\n下载完成! 成功下载 {success_count}/{len(files)} 个文件")
        
    def load_training_data(self):
        """加载训练数据"""
        print("\n加载训练数据...")
        
        try:
            # 读取中文和英文句子
            with open(self.data_dir / "TM-training-set/chinese.txt", 'r', encoding='utf-8') as f:
                chinese_sentences = [line.strip() for line in f.readlines()]
            
            with open(self.data_dir / "TM-training-set/english.txt", 'r', encoding='utf-8') as f:
                english_sentences = [line.strip() for line in f.readlines()]
            
            # 读取对齐信息
            with open(self.data_dir / "TM-training-set/Alignment.txt", 'r', encoding='utf-8') as f:
                alignments = [line.strip() for line in f.readlines()]
            
            # 创建DataFrame
            training_data = pd.DataFrame({
                'chinese': chinese_sentences,
                'english': english_sentences,
                'alignment': alignments
            })
            
            print(f"训练数据加载完成，共 {len(training_data)} 个句子对")
            return training_data
            
        except Exception as e:
            print(f"加载训练数据失败: {e}")
            return None
    
    def load_dev_data(self):
        """加载开发集数据"""
        print("\n加载开发集数据...")
        
        try:
            with open(self.data_dir / "Dev-set/Niu.dev.txt", 'r', encoding='utf-8') as f:
                dev_data = [line.strip() for line in f.readlines()]
            
            print(f"开发集数据加载完成，共 {len(dev_data)} 个句子")
            return dev_data
            
        except Exception as e:
            print(f"加载开发集数据失败: {e}")
            return None
    
    def load_test_data(self):
        """加载测试数据"""
        print("\n加载测试数据...")
        
        try:
            with open(self.data_dir / "Test-set/Niu.test.txt", 'r', encoding='utf-8') as f:
                test_data = [line.strip() for line in f.readlines()]
            
            print(f"测试数据加载完成，共 {len(test_data)} 个句子")
            return test_data
            
        except Exception as e:
            print(f"加载测试数据失败: {e}")
            return None
    
    def load_reference_data(self):
        """加载参考翻译数据"""
        print("\n加载参考翻译数据...")
        
        try:
            with open(self.data_dir / "reference-set/Niu.test.reference", 'r', encoding='utf-8') as f:
                reference_data = [line.strip() for line in f.readlines()]
            
            print(f"参考翻译数据加载完成，共 {len(reference_data)} 个句子")
            return reference_data
            
        except Exception as e:
            print(f"加载参考翻译数据失败: {e}")
            return None
    
    def parse_alignment(self, alignment_str, chinese_sentence, english_sentence):
        """解析对齐信息"""
        chinese_words = chinese_sentence.split()
        english_words = english_sentence.split()
        
        align_pairs = []
        for pair in alignment_str.split():
            try:
                ch_idx, en_idx = map(int, pair.split('-'))
                if ch_idx < len(chinese_words) and en_idx < len(english_words):
                    align_pairs.append({
                        'chinese_word': chinese_words[ch_idx],
                        'english_word': english_words[en_idx],
                        'ch_index': ch_idx,
                        'en_index': en_idx
                    })
            except ValueError:
                continue
        
        return align_pairs
    
    def get_sentence_with_alignment(self, index):
        """获取指定索引的句子及其对齐信息"""
        try:
            training_data = self.load_training_data()
            if training_data is None or index >= len(training_data):
                print(f"索引 {index} 超出范围")
                return None
            
            row = training_data.iloc[index]
            alignment_info = self.parse_alignment(
                row['alignment'], 
                row['chinese'], 
                row['english']
            )
            
            return {
                'chinese': row['chinese'],
                'english': row['english'],
                'alignment': alignment_info,
                'alignment_raw': row['alignment']
            }
            
        except Exception as e:
            print(f"获取句子对齐信息失败: {e}")
            return None
    
    def show_dataset_info(self):
        """显示数据集信息"""
        print("\n" + "="*50)
        print("NiuTrans 数据集信息")
        print("="*50)
        
        # 检查文件是否存在并统计
        files_info = [
            ("TM-training-set/chinese.txt", "训练集中文"),
            ("TM-training-set/english.txt", "训练集英文"), 
            ("TM-training-set/Alignment.txt", "对齐信息"),
            ("Dev-set/Niu.dev.txt", "开发集"),
            ("Test-set/Niu.test.txt", "测试集"),
            ("reference-set/Niu.test.reference", "参考翻译")
        ]
        
        for file_path, description in files_info:
            full_path = self.data_dir / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    print(f"✓ {description}: {line_count} 行")
                except:
                    print(f"✓ {description}: 文件存在")
            else:
                print(f"✗ {description}: 文件缺失")

def main():
    # 创建数据集实例
    dataset = NiuTransDataset()
    
    # 下载数据集
    dataset.download_dataset()
    
    # 显示数据集信息
    dataset.show_dataset_info()
    
    # 加载数据
    training_data = dataset.load_training_data()
    dev_data = dataset.load_dev_data()
    test_data = dataset.load_test_data()
    reference_data = dataset.load_reference_data()
    
    # 示例：显示第105个句子的对齐信息（如描述中的例子）
    if training_data is not None and len(training_data) > 105:
        print("\n" + "="*50)
        print("示例：第105个句子的对齐信息")
        print("="*50)
        
        example = dataset.get_sentence_with_alignment(104)  # 索引从0开始
        if example:
            print(f"中文: {example['chinese']}")
            print(f"英文: {example['english']}")
            print(f"对齐信息: {example['alignment_raw']}")
            print("\n详细对齐:")
            for align in example['alignment']:
                print(f"  '{align['chinese_word']}' -> '{align['english_word']}'")
    
    print("\n数据集准备完成！")
    print(f"数据保存在: {dataset.data_dir.absolute()}")

if __name__ == "__main__":
    # 安装依赖（如果还没有安装）
    try:
        import requests
        import pandas
    except ImportError:
        print("请安装所需依赖:")
        print("pip install requests pandas")
        exit(1)
    
    main()