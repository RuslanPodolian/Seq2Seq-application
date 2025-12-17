from fileinput import filename
import torch.nn as nn
import torch

if __name__ == "__main__":
    word2idx = {}
    idx2word = {}
    idx = 0

    char = "How are you, bro?"

    for word in char.split():
        if word not in word2idx:
            word2idx[word] = idx
            idx2word[idx] = word
            idx += 1

    print(word2idx)
    print(idx2word)

    char2 = "How are you, bro?"
    seq = []

    for word in char2.split():
        seq.append(word2idx[word])

    seq = torch.tensor(seq)
    print(seq)

    embedding = nn.Embedding(len(word2idx), 10)

    embedded_seq = embedding(seq)

    print(embedded_seq)

    import fnmatch

    file_name = 'hello.tsv-00000-of-00001'
    if fnmatch.fnmatch(file_name, '*.tsv*'):
        print("File is a TSV file")
    else:
        print("File is not a TSV file")

    seq = torch.tensor([1, 2, 3, 4, 5])
    x = seq[:-1]
    y = seq[-1]
    print(x)
    print(y)