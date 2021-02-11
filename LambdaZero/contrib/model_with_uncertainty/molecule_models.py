import sys, time

from torch.utils.data import DataLoader
from LambdaZero.models.torch_graph_models import MPNNet_Parametric, fast_from_data_list
from LambdaZero.models import MPNNetDrop
from LambdaZero.contrib.inputs import ListGraphDataset
from .model_with_uncertainty import ModelWithUncertainty



class MolFP(ModelWithUncertainty):
    def __init__(self):
        ModelWithUncertainty.__init__(self)
        self.model = MPNNetDrop(True, False, True, 0.1, 72) # fixme 72 features is insane!!
        # drop_last, drop_data, drop_weights, drop_prob, num_feat=14

    def fit(self,x,y):
        graphs = [m["mol_graph"] for m in x]
        dataset = ListGraphDataset(graphs)
        dataloader = DataLoader(dataset, batch_size=2,collate_fn=fast_from_data_list)

        for batch in dataloader:
            print("batch", batch, self.model.forward(batch, do_dropout=True))

        #print("exps:", "%.3f" % (self.num_fit / (time.time() - self.start)))


    def get_mean_and_variance(self,x):
        # todo:
        #print("mean variance not implemented", x)
        mean = [0.1 for i in x]
        var = [0.2 for i in x]
        return mean, var