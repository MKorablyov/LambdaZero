import time
import numpy as np
from LambdaZero.contrib.proxy import Actor
from random import random
from LambdaZero.contrib.oracle import QEDOracle, SynthOracle
from LambdaZero.environments.block_mol_v3 import synth_config
import ray
import LambdaZero.contrib.functional

import os
import pandas as pd
import torch
from torch import nn
from rdkit import rdBase
from rdkit.Chem import MolFromSmiles
import selfies as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAEEncoder(nn.Module):

    def __init__(self, in_dimension, layer_1d, layer_2d, layer_3d,
                 latent_dimension):
        super(VAEEncoder, self).__init__()
        self.latent_dimension = latent_dimension

        self.encode_nn = nn.Sequential(
            nn.Linear(in_dimension, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, layer_2d),
            nn.ReLU(),
            nn.Linear(layer_2d, layer_3d),
            nn.ReLU()
        )

        self.encode_mu = nn.Linear(layer_3d, latent_dimension)
        self.encode_log_var = nn.Linear(layer_3d, latent_dimension)

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        h1 = self.encode_nn(x)
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class VAEDecoder(nn.Module):

    def __init__(self, latent_dimension, gru_stack_size, gru_neurons_num,
                 out_dimension):
        super(VAEDecoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num

        self.decode_RNN = nn.GRU(
            input_size=latent_dimension,
            hidden_size=gru_neurons_num,
            num_layers=gru_stack_size,
            batch_first=False)

        self.decode_FC = nn.Sequential(
            nn.Linear(gru_neurons_num, out_dimension),
        )

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_stack_size, batch_size,
                                self.gru_neurons_num)

    def forward(self, z, hidden):
        l1, hidden = self.decode_RNN(z, hidden)
        decoded = self.decode_FC(l1)  # fully connected layer

        return decoded, hidden




class DiversityReward:
    def __init__(self,
                 encoder_model,
                 train=False,
                 file_name_smiles=None,
                 dataset=None,
                 data_train=None,
                 data_valid=None,
                 num_epochs=500,
                 batch_size=32,
                 lr_enc=1e-3,
                 lr_dec=1e-3,
                 KLD_alpha=0.1):

        self.pretrained_encoder_model = encoder_model
        self.vae_encoder = VAEEncoder()
        self.vae_decoder = VAEDecoder()
        self.data_train = data_train
        self.data_valid = data_valid
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr_enc = lr_enc
        self.lr_dec = lr_dec
        self.KLD_alpha = KLD_alpha
        self.file_name_smiles = file_name_smiles
        self.encoding_list = None
        self.encoding_alphabet = None
        self.largest_molecule_len = None
        self.len_max_molec = None
        self.len_alphabet = None
        self.len_max_mol_one_hot = None
        self.data_train = None
        self.data_valid = None
        if train:
            self.start_training()

    def start_training(self):
        self.encoding_list, self.encoding_alphabet, self.largest_molecule_len, _, _, _ = \
            self.get_selfie_and_smiles_encodings_for_dataset(self.file_name_smiles)

        print('--> Creating one-hot encoding...')
        data = self.multiple_selfies_to_hot()
        print('Finished creating one-hot encoding.')
        self.len_max_molec = data.shape[1]
        self.len_alphabet = data.shape[2]
        self.len_max_mol_one_hot = self.len_max_molec * self.len_alphabet

        print(' ')
        print(f"Alphabet has {self.len_alphabet} letters, "
              f"largest molecule is {self.len_max_molec} letters.")

        batch_size = self.batch_size

        self.vae_encoder = VAEEncoder(in_dimension=self.len_max_mol_one_hot).to(device)
        self.vae_decoder = VAEDecoder(out_dimension=len(self.encoding_alphabet)).to(device)

        print('*' * 15, ': -->', device)

        data = torch.tensor(data, dtype=torch.float).to(device)

        train_valid_test_size = [0.5, 0.5, 0.0]
        data = data[torch.randperm(data.size()[0])]
        idx_train_val = int(len(data) * train_valid_test_size[0])
        idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])

        self.data_train = data[0:idx_train_val]
        self.data_valid = data[idx_train_val:idx_val_test]

        print("start training")
        self.train_model()

    def reset(self, previous_reward=0.0):
        return None

    def train_model(self):
        """
        Train the Variational Auto-Encoder
        """

        print('num_epochs: ', self.num_epochs)

        # initialize an instance of the model
        optimizer_encoder = torch.optim.Adam(self.vae_encoder.parameters(), lr=self.lr_enc)
        optimizer_decoder = torch.optim.Adam(self.vae_decoder.parameters(), lr=self.lr_dec)

        data_train = self.data_train.clone().detach().to(device)
        num_batches_train = int(len(data_train) / self.batch_size)

        quality_valid_list = [0, 0, 0, 0]
        for epoch in range(self.num_epochs):

            data_train = data_train[torch.randperm(data_train.size()[0])]

            start = time.time()
            for batch_iteration in range(num_batches_train):  # batch iterator

                # manual batch iterations
                start_idx = batch_iteration * self.batch_size
                stop_idx = (batch_iteration + 1) * self.batch_size
                batch = data_train[start_idx: stop_idx]

                # reshaping for efficient parallelization
                inp_flat_one_hot = batch.flatten(start_dim=1)
                latent_points, mus, log_vars = self.vae_encoder(inp_flat_one_hot)

                latent_points = latent_points.unsqueeze(0)
                hidden = self.vae_decoder.init_hidden(batch_size=self.batch_size)

                out_one_hot = torch.zeros_like(batch, device=device)
                for seq_index in range(batch.shape[1]):
                    out_one_hot_line, hidden = self.vae_decoder(latent_points, hidden)
                    out_one_hot[:, seq_index, :] = out_one_hot_line[0]

                # compute ELBO
                loss = self.compute_elbo(batch, out_one_hot, mus, log_vars)

                # perform back propogation
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.vae_decoder.parameters(), 0.5)
                optimizer_encoder.step()
                optimizer_decoder.step()

                if batch_iteration % 30 == 0:
                    end = time.time()

                    # assess reconstruction quality
                    quality_train = self.compute_recon_quality(batch, out_one_hot)
                    quality_valid = self.quality_in_valid_set(self.vae_encoder, self.vae_decoder,
                                                         self.data_valid, self.batch_size)

                    report = 'Epoch: %d,  Batch: %d / %d,\t(loss: %.4f\t| ' \
                             'quality: %.4f | quality_valid: %.4f)\t' \
                             'ELAPSED TIME: %.5f' \
                             % (epoch, batch_iteration, num_batches_train,
                                loss.item(), quality_train, quality_valid,
                                end - start)
                    print(report)
                    start = time.time()

            if epoch % 10 == 0:
                self.save_models(self.vae_encoder, self.vae_decoder, epoch)

    def compute_elbo(self, x, x_hat, mus, log_vars):
        inp = x_hat.reshape(-1, x_hat.shape[2])
        target = x.reshape(-1, x.shape[2]).argmax(1)

        criterion = torch.nn.CrossEntropyLoss()
        recon_loss = criterion(inp, target)
        kld = -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())

        return recon_loss + self.KLD_alpha * kld

    def compute_recon_quality(self, x, x_hat):
        x_indices = x.reshape(-1, x.shape[2]).argmax(1)
        x_hat_indices = x_hat.reshape(-1, x_hat.shape[2]).argmax(1)

        differences = 1. - torch.abs(x_hat_indices - x_indices)
        differences = torch.clamp(differences, min=0., max=1.).double()
        quality = 100. * torch.mean(differences)
        quality = quality.detach().cpu().numpy()

        return quality

    def quality_in_valid_set(self):
        data_valid = self.data_valid[torch.randperm(self.data_valid.size()[0])]  # shuffle
        num_batches_valid = len(data_valid) // self.batch_size

        quality_list = []
        for batch_iteration in range(min(25, num_batches_valid)):

            # get batch
            start_idx = batch_iteration * self.batch_size
            stop_idx = (batch_iteration + 1) * self.batch_size
            batch = data_valid[start_idx: stop_idx]
            _, trg_len, _ = batch.size()

            inp_flat_one_hot = batch.flatten(start_dim=1)
            latent_points, mus, log_vars = self.vae_encoder(inp_flat_one_hot)

            latent_points = latent_points.unsqueeze(0)
            hidden = self.vae_decoder.init_hidden(batch_size=self.batch_size)
            out_one_hot = torch.zeros_like(batch, device=device)
            for seq_index in range(trg_len):
                out_one_hot_line, hidden = self.vae_decoder(latent_points, hidden)
                out_one_hot[:, seq_index, :] = out_one_hot_line[0]

            # assess reconstruction quality
            quality = self.compute_recon_quality(batch, out_one_hot)
            quality_list.append(quality)

        return np.mean(quality_list).item()




    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        return self.eval(molecule)

    def selfies_to_hot(self, selfie, largest_selfie_len, alphabet):
        symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))
        selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))
        symbol_list = sf.split_selfies(selfie)
        integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]
        onehot_encoded = list()
        for index in integer_encoded:
            letter = [0] * len(alphabet)
            letter[index] = 1
            onehot_encoded.append(letter)

        return integer_encoded, np.array(onehot_encoded)

    def multiple_selfies_to_hot(self, selfies_list, largest_molecule_len, alphabet):
        """Convert a list of selfies strings to a one-hot encoding
        """

        hot_list = []
        for s in selfies_list:
            _, onehot_encoded = self.selfies_to_hot(s, largest_molecule_len, alphabet)
            hot_list.append(onehot_encoded)
        return np.array(hot_list)

    def _make_dir(self, directory):
        os.makedirs(directory)

    def save_models(self, encoder, decoder, epoch):
        out_dir = './saved_models/{}'.format(epoch)
        self._make_dir(out_dir)
        torch.save(encoder.state_dict(), '{}/E.pt'.format(out_dir))
        torch.save(decoder.state_dict(), '{}/D.pt'.format(out_dir))

    def sample_latent_space(self, vae_encoder, vae_decoder, sample_len):
        vae_encoder.eval()
        vae_decoder.eval()

        gathered_atoms = []

        fancy_latent_point = torch.randn(1, 1, vae_encoder.latent_dimension,
                                         device=device)
        hidden = vae_decoder.init_hidden()

        # runs over letters from molecules (len=size of largest molecule)
        for _ in range(sample_len):
            out_one_hot, hidden = vae_decoder(fancy_latent_point, hidden)

            out_one_hot = out_one_hot.flatten().detach()
            soft = nn.Softmax(0)
            out_one_hot = soft(out_one_hot)

            out_index = out_one_hot.argmax(0)
            gathered_atoms.append(out_index.data.cpu().tolist())

        vae_encoder.train()
        vae_decoder.train()

        return gathered_atoms

    def get_selfie_and_smiles_encodings_for_dataset(file_path):
        df = pd.read_csv(file_path)

        smiles_list = np.asanyarray(df.smiles)
        print("smiles list shape = ", smiles_list.shape)

        smiles_alphabet = list(set(''.join(smiles_list)))
        smiles_alphabet.append(' ')  # for padding

        largest_smiles_len = len(max(smiles_list, key=len))

        print('--> Translating SMILES to SELFIES...')
        selfies_list = list(map(sf.encoder, smiles_list))

        all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
        all_selfies_symbols.add('[nop]')
        selfies_alphabet = list(all_selfies_symbols)

        largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

        print('Finished translating SMILES to SELFIES.')

        return selfies_list, selfies_alphabet, largest_selfies_len, \
               smiles_list, smiles_alphabet, largest_smiles_len

