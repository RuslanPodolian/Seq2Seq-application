import torch
import os
from torch.utils.data import Dataset, DataLoader
import fnmatch
import json

vocab_path = 'data/vocabulary.json'
config_path = 'data/config.json'

class JSONConfig:
    def __init__(self, path):
        self.path = path

    def get_value(self, key):
        with open(self.path, 'r') as f:
            json_data = json.load(f)

            return json_data[key]

    def change_value(self, key, value):
        with open(self.path, 'r') as f:
            json_data = json.load(f)
        
        json_data[key] = value

        with open(self.path, 'w') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        return json_data

    def save_dict_to_json(self, path, dict):
        with open(path, 'a', encoding='utf-8') as f:
            json.dump(dict, f, ensure_ascii=False, indent=4)
        
        return dict

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.json_config = JSONConfig(path=vocab_path)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        return self.word2idx[word]

    def remove_word(self, word):
        if word in self.word2idx:
            del self.word2idx[word]
            del self.idx2word[self.word2idx[word]]
            self.idx -= 1
        return self.word2idx[word]

    def save_vocab_to_json(self, path):
        self.json_config.save_dict_to_json(path, {
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'idx': self.idx,
            })

        self.json_config.change_value('vocabulary_saved', True)

        return self.json_config.get_value('vocabulary_saved')

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, idx):
        return self.idx2word[idx]


class CorpusDataset(Dataset):
    def __init__(self):
        self.vocabulary = Vocabulary()
        self.data = []
        self.x = []
        self.y = []

    def get_data(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                words = line.strip().split()

                for word in words:
                    self.vocabulary.add_word(word)
        
        return self.vocabulary
    
    def set_data(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                words = line.strip().split()

                seq = []

                for word in words:
                    seq.append(self.vocabulary.add_word(word))

                seq = torch.tensor(seq)

                seq = seq.unsqueeze(1)

                self.data.append(seq)
        
        return self.data
    
    def next_word_prediction_separator(self, data):
        for seq in data:
            self.x.append(seq[:-1])
            self.y.append(seq[1:])

        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)

        return self.x, self.y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.corpus.next_word_prediction_separator(self.corpus.data)

        return x[idx], y[idx]

class DatasetConfig:
    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.corpus = CorpusDataset()
        self.json_config = JSONConfig(path=config_path)

    def go_through_text(self, file_path):
        self.corpus.set_data(file_path)

        return self.corpus.data

    def set_loader(self):
        return DataLoader(self.corpus, batch_size=self.batch_size, shuffle=self.shuffle)

    def go_through_dir(self, dir_path):
        for file in os.listdir(dir_path):
            try:
                if file.endswith(".txt") or fnmatch.fnmatch(file, '*.tsv*'):
                    self.go_through_text(os.path.join(dir_path, file))
                
                print(f"Scrolling through file {file}: Completed")
            except Exception as e:
                print(f"Error scrolling through file {file}: {e}")
                continue
        
        self.json_config.change_value('dataset_scrolled', True)

        print("Dataset scrolled through")
        # return self.corpus.data

    def complete_scrolling(self, dataset_path):
        print("Scrolling through dataset...")
        self.go_through_dir(dataset_path)

        ret = config.corpus.vocabulary.save_vocab_to_json(vocab_path)
        
        print("Saved vocabulary to JSON: ", ret)
        
        print("Dataset scrolled through")

        return ret


    def get_main_loader(self):
        main_loader = DataLoader(self.corpus, batch_size=self.batch_size, shuffle=self.shuffle)
     
        return main_loader
        

if __name__ == "__main__":
    print("Starting dataset configuration...")
    config = DatasetConfig()
    dataset_path = 'dataset'
    print("Dataset scrolled: ", JSONConfig(path=config_path).get_value('dataset_scrolled'))
    if JSONConfig(path=config_path).get_value('dataset_scrolled') == False:
        config.complete_scrolling(dataset_path)
    
    loader = config.get_main_loader()

    batch = next(iter(loader))

    print(batch)