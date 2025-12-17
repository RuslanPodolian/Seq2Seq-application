from model.model import EncoderDecoder
import torch
import torch.optim as optim
import torch.nn as nn

class Trainer:
    def __init__(self, model, train_loader, val_loader, batch_size, shuffle=True, num_epochs=100):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def train(self):
        self.model.train()

        for epoch in range(num_epochs):
