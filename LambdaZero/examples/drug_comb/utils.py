from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch


def get_project_root():
    return Path(__file__).parent.parent


def get_dropout_modules_recursive(module):
    '''
    A generator for retrieving dropout modules within an
    instance of a subclass of torch.nn.Module.

    Arguments
    ---------
    modules : Iterator[torch.nn.Module]
        An iterable object consisting only of instances whose
        types subclass torch.nn.Module

    Yields
    ------
    Iterator[torch.nn.Module]
        An iterator of dropout modules which belong to the
        modules in the argument modules.

    Throws
    ------
    AssertionError
        Throws AssertionError if a module in the modules
        argument is not an instance of a subclass of
        torch.nn.Module.
    '''
    for child_module in module.modules():
        if module.__class__.__name__.startswith('Dropout'):
            yield module

        if child_module != module:
            yield from get_dropout_modules_recursive(child_module)


class MeanVarianceNormalizer:
    def __init__(self, responses):
        self.mean = responses.mean()
        self.std = responses.std().item()

    def tfm(self, x):
        """
        Normalize
        """
        x_norm = (x - self.mean) / self.std
        return x_norm

    def itfm(self, x_norm):
        """
        Unnormalize
        """
        x = (x_norm * self.std) + self.mean
        return x

class IntParamRegistrar:
    _instance = None

    class _IntParamRegistrar:
        def __init__(self):
            self.registered_args = set()

        def register_arg(self, arg):
            self.registered_args.add(arg)

        def get_registered_args(self):
            return self.registered_args

    @staticmethod
    def _get_instance():
        """
        How this method looks with the IntParamRegistrars is
        why I hate python's staticmethod
        """
        if IntParamRegistrar._instance is None:
            IntParamRegistrar._instance = IntParamRegistrar._IntParamRegistrar()

        return IntParamRegistrar._instance

    @staticmethod
    def register_arg(arg):
        IntParamRegistrar._get_instance().register_arg(arg)

    @staticmethod
    def register_args(args):
        for arg in args:
            IntParamRegistrar._get_instance().register_arg(arg)

    @staticmethod
    def get_registered_args():
        return IntParamRegistrar._get_instance().get_registered_args()

    @staticmethod
    def register_hp_best_params(best_params, exceptions=set()):
        # Should be a list of length 1
        assert isinstance(best_params, list) and len(best_params) == 1

        exceptions = set(exceptions)
        for argname, value in best_params[0].items():
            if isinstance(value, int) and argname not in exceptions:
                IntParamRegistrar.register_arg(argname)

def get_fingerprint(smile, radius, n_bits):
    if smile == 'none':
        return np.array([-1] * n_bits)
    try:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius, n_bits))
    except Exception as ex:
        return np.array([-1] * n_bits)

def to_zero_base(v):
    bins = np.unique(v.cpu().flatten()) + 1
    return torch.from_numpy(np.digitize(v.cpu(), bins)).to(v.device)

