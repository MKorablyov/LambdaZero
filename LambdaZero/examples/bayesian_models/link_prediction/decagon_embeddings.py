import numpy as np
import os, time, os.path as osp
import torch as th
import sklearn
import ray
from ray import tune

from torch.utils.data import DataLoader, Dataset, Subset
from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


def load_decagon(dataset_root, relation):
    df = np.load(osp.join(dataset_root, relation + ".npz"))
    return df["Z"], df["D"], df["R"], df["labels"], df["is_train"]

class DecagonEmbeddingDataset(Dataset):
    def __init__(self, dataset_root, relation):
        self.Z, self.D, self.R, self.labels, _ = load_decagon(dataset_root,relation)

    def __getitem__(self, index):
        index2d = (index // self.Z.shape[0], index % self.Z.shape[0])
        return self.Z[index2d[0]], self.Z[index2d[1]], self.labels[index]

    def __len__(self):
        return self.Z.shape[0] * self.Z.shape[0]

class LinearDec(th.nn.Module):
    def __init__(self, Z, R, D):
        super(LinearDec, self).__init__()
        self.Z = Z
        self.R = R
        self.D = th.nn.Parameter(th.empty(32,32).normal_(mean=0,std=0.01), requires_grad=True)

    def forward(self, Zi, Zj):
        logits = (Zi @ self.D @ self.R @ self.D * Zj).sum(1)
        return logits



def train_epoch(loader, model, optimizer, device, config):
    model.train()
    labels_all = []
    preds_all = []
    loss_all = []
    for bidx, (Zi, Zj, labels) in enumerate(loader):
        optimizer.zero_grad()
        Zi, Zj, labels = Zi.to(device), Zj.to(device), labels.to(device)
        logits = model(Zi, Zj)
        # train step
        # Q = sigmoid(logits)
        # P = probs

        # CE = -P log(Q)
        #P = th.FloatTensor(labels.shape[0], 2)
        #P.zero_()
        #P.scatter_(th.LongTensor(1), labels, th.LongTensor(1))
        #print("P", P)

        loss = ((logits - labels)**2).sum()
        # true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
        # negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
        # loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)

        loss.backward()
        optimizer.step()

        labels_all.append(labels.cpu().numpy())
        preds_all.append(th.nn.Sigmoid()(logits).detach().cpu().numpy())
        loss_all.append(loss.detach().cpu().numpy())


    labels_all, preds_all = np.concatenate(labels_all, axis=0), np.concatenate(preds_all, axis=0)
    auc = sklearn.metrics.roc_auc_score(labels_all, preds_all)

    print("train", auc, len(labels_all), sum(labels_all))
    return {"AUC": auc}


def eval_epoch(loader, model, device, config):
    model.eval()
    labels_all = []
    preds_all = []
    for bidx, (Zi, Zj, labels) in enumerate(loader):
        Zi, Zj, labels = Zi.to(device), Zj.to(device), labels.to(device)
        logits = model(Zi, Zj)
        labels_all.append(labels.cpu().numpy())
        preds_all.append(th.nn.Sigmoid()(logits).detach().cpu().numpy())

    labels_all, preds_all = np.concatenate(labels_all, axis=0), np.concatenate(preds_all, axis=0)
    auc = sklearn.metrics.roc_auc_score(labels_all, preds_all)

    #confusion = sklearn.metrics.confusion_matrix(labels_all, preds_all)


    print("test", auc, len(labels_all), sum(labels_all))
    return {"AUC": auc}


class BasicRegressor(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        # load dataset
        dataset = DecagonEmbeddingDataset(config["dataset_root"], config["relation"])

        # split dataset
        Z, D, R, _, is_train = load_decagon(config["dataset_root"], config["relation"])
        train_set = Subset(dataset, np.where(is_train)[0][10:13]) # fixme
        self.train_set = th.utils.data.DataLoader(train_set, batch_size=config["b_size"], shuffle=True)
        test_set = Subset(dataset, np.where(~is_train)[0])
        self.test_set = th.utils.data.DataLoader(test_set, batch_size=config["b_size"])

        # make model
        self.model = LinearDec(Z=th.tensor(Z).to(self.device),
                               R=th.tensor(R).to(self.device),
                               D=th.tensor(D).to(self.device))
        self.model.to(self.device)
        self.optim = th.optim.Adam(self.model.parameters(), lr=config["lr"])

        # make epochs
        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _train(self):
        train_scores = self.train_epoch(self.train_set, self.model, self.optim, self.device, self.config)
        eval_scores = self.eval_epoch(self.test_set, self.model,  self.device, self.config)
        # rename to make scope
        train_scores = [("train_" + k, v) for k,v in train_scores.items()]
        eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
        scores = dict(train_scores + eval_scores)
        return scores

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        th.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(th.load(checkpoint_path))





relations = ['C0003126','C0020456','C0027947', 'C0026780', 'C0009193', 'C0038019' ]

DEFAULT_CONFIG = {
    "trainer": BasicRegressor,
    "trainer_config": {
        "relation": relations[0],
        "dataset_root": os.path.join(datasets_dir, "decagon_embeddings"),
        "lr": 0.001,
        "b_size": 64,
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        },
    "summaries_dir": summaries_dir,
    "memory": 20 * 10 ** 9,
    "checkpoint_freq": 250000000,
    "stop": {"training_iteration": 2},
}

config = DEFAULT_CONFIG

if __name__ == "__main__":
    ray.init()

    analysis = tune.run(config["trainer"],
                        config=config["trainer_config"],
                        stop={"training_iteration":100}, #EarlyStop(),
                        resources_per_trial={
                           "cpu": 4, # fixme requesting all CPUs blocks additional call to ray from LambdaZero.input
                           "gpu": 1.0
                        },
                        num_samples=1,
                        checkpoint_at_end=False,
                        local_dir=summaries_dir,
                        checkpoint_freq=100000)




# dataset = DecagonEmbeddingDataset()
# _, _, _, _, is_test = load_decagon()
# test_set = Subset(dataset, np.where(~is_test)[0])
# test_set = th.utils.data.DataLoader(test_set, batch_size=902, shuffle=True)
# for (Zi, Zj, labels) in test_set:
#     D, R = th.tensor(dataset.D), th.tensor(dataset.R)
#     preds = (Zi @ D @ R @ D * Zj).sum(1)
#     print("AUC", sklearn.metrics.roc_auc_score(labels, preds))