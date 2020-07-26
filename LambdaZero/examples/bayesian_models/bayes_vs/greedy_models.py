
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import chem_ops
import torch.nn.functional as F

class cfg:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 500
    epochs = 6
    std = 0.5
    mean = 0.
    mode = 'y'
    place_to_save_model = 'place_to_save_model.pt'

class CustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data)) 
        # print(transposed_data)
        self.x = torch.stack(transposed_data[0], 0)
        self.y = torch.stack(transposed_data[1], 0)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 1))

    def forward(self, x):
        fc1 = self.fc1(x)
        return self.fc2(fc1)

class Trainer:
    def __init__(self, model):
        self.model = model
        self.variance_list = []

    def create_dataloader(self, x, y, batch_size = 500):
        
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size = batch_size, collate_fn = self.collate_wrapper)
        return loader

    def train(self, x_train, y_train, batch_size = 500):
        try:
            self.load_model()
        except:
            pass
        model = self.model
        train_loader = self.create_dataloader(x_train, y_train, batch_size)
        optimizer = torch.optim.Adam(model.parameters())
        model.train()
        for _ in range(cfg.epochs):
            for _, batch in enumerate(train_loader):
                optimizer.zero_grad()
                pred = model(batch.x)
                loss = F.mse_loss(pred.squeeze(), batch.y)
                loss.backward()
                optimizer.step()
        self.save_model()


    def predict(self, x_test):
        model = self.model
        model.eval()
        return model(x_test)

    def collate_wrapper(self, batch):
        return CustomBatch(batch)

    def save_model(self):
        torch.save(self.model.state_dict(), cfg.place_to_save_model)

    def load_model(self):
        self.model.load_state_dict(torch.load(cfg.place_to_save_model))


