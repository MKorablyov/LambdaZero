import os.path as osp
import torch
from LambdaZero.examples.baselines.guacamol_baseline.smiles_lstm_hc.rnn_utils import load_rnn_model, load_smiles_from_list, get_tensor_dataset
# from LambdaZero.utils import get_external_dirs
# datasets_dir, programs_dir, summaries_dir = get_external_dirs()
datasets_dir = "/home/nekoeiha/LambdaZero_deup/LambdaZero/examples/baselines/"


def probability_for_batch(smiles_list, device):
  model = load_rnn_model(model_definition=osp.join(datasets_dir, 'guacamol_baseline/smiles_lstm_hc/pretrained_model',
                                                   'model_final_0.473.json'),
                         model_weights=osp.join(datasets_dir, 'guacamol_baseline/smiles_lstm_hc/pretrained_model',
                                                'model_final_0.473.pt'), device=device)

  o= load_smiles_from_list(smiles_list) # may want to convert all to mol and back to make sure it's canonical
  batch = get_tensor_dataset(o[0])
  XXX = batch[0:][0].to(device)
  YYY = batch[0:][1].to(device)
  # import pdb;
  # pdb.set_trace()
  i=torch.nonzero(XXX).max()
  XXX = XXX[:,0:i+2]
  YYY = YYY[:,0:i+2]
  h = model.init_hidden(XXX.size(0), device=device)
  y, _ = model.forward(XXX, h)
  p = torch.nn.functional.softmax(y, dim=2)
  # here you still need to use the next char ground truth (YYY) to look up the probabs
  # in p, but COLAB is giving me a hard time and crashes
  # return p
  return torch.max(p, dim=2).values.log().sum(dim=1), o[1]


# print(probability_for_batch(model, ['CCCCCC', 'NNNNNNNC=O']))