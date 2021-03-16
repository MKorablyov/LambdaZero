import sys, os, os.path as osp

import torch
from tqdm import tqdm
import wandb

from LambdaZero.examples.mpnn.dime_net.qm9 import QM9
from LambdaZero.examples.mpnn.dime_net.dataloader import DataLoader
from LambdaZero.models.torch_models import MPNNet
from LambdaZero.examples.egnn import qm9_config
from LambdaZero.examples.egnn.train_egnn_qm9 import MyTransform

from LambdaZero.utils import get_external_dirs
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

if len(sys.argv) >= 2: config_name = sys.argv[1]
else: config_name = "qm9_1k"
config = getattr(qm9_config,config_name)

if config["dry_run"]:
    os.environ["WANDB_MODE"] = "dryrun"
with open(osp.join(summaries_dir, "wandb_key"), 'r') as file:
    key = file.read().replace('\n', '')
os.environ["WANDB_API_KEY"] = key
os.environ["WANDB_DIR"] = summaries_dir
wandb.init(project="egnn")
wandb.run.name = 'MPNN' + str(config_name) #wandb.run.id

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
dataset = QM9(path, transform=MyTransform()).shuffle()

if config["train_dataset_size"]:
    train_dataset = dataset[:int(config["train_dataset_size"]*130000)]
    val_dataset = dataset[int(config["train_dataset_size"]*130000):int(config["train_dataset_size"]*1.15*130000)]
    test_dataset = dataset[int(config["train_dataset_size"]*1.15*130000):int(config["train_dataset_size"]*1.30*130000)]
else:
    train_dataset = dataset[2:1000]
    val_dataset = dataset[1000:2000]
    test_dataset = dataset[2000:3000]

train_loader = DataLoader(train_dataset, 96, shuffle=True, num_workers=6)
val_loader = DataLoader(val_dataset, 96, num_workers=6)
test_loader = DataLoader(test_dataset, 96, num_workers=6)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if config["train_dataset_size"]:
    model = MPNNet(num_feat=dataset.num_node_features, dim=64).to(device)
else:
    model = MPNNet(num_feat=dataset.num_node_features, dim=128).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-16) #amsgrad=True) #

if config["scheduler"]:
    scheduler = config["scheduler"](optimizer, T_max=300)


def train(loader):
    model.train()
    total_loss = 0
    pbar = tqdm(total=len(loader))
    for data in loader:

        optimizer.zero_grad()
        if config["control_exp"]:
            data.pos = torch.randn(list(data.pos.shape)[0], list(data.pos.shape)[1])
        data = data.to(device)

        out = model(data)
        loss = (out - data.y).abs().mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pbar.set_description(f'Loss: {loss:.4f}')
        pbar.update()
    pbar.close()
    if config["scheduler"]:
        scheduler.step()

    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_mae = 0
    for data in loader:
        if config["control_exp"]:
            data.pos = torch.randn(list(data.pos.shape)[0], list(data.pos.shape)[1])
        data = data.to(device)
        out = model(data)
        total_mae += (out - data.y).abs().sum().item()

    return total_mae / len(loader.dataset)


best_val_mae = test_mae = float('inf')
for epoch in range(1, config["training_iterations"]+1):
    train_mae = train(train_loader)
    val_mae = test(val_loader)
    test_mae = test(test_loader)
    wandb.log({'epoch': epoch, 'train_mae': train_mae, "val_mae": val_mae, "test_mae": test_mae})
    print(f'Epoch: {epoch:02d}, Train: {train_mae:.4f}, Val: {val_mae:.4f}, '
          f'Test: {test_mae:.4f}')