import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from LambdaZero.environments import MolMDP


@pytest.fixture
def debug_blocks():
    """
    A subset of the blocks_PDB_105.json file for testing purposes.
    """
    debug_blocks = {
        "block_name": {
            "0": "c1ccccc1_0",
            "1": "CO_0",
            "2": "CO_1",
            "3": "C=O_0",
            "4": "C1CCNCC1_0",
            "5": "C1CCNCC1_1",
            "6": "C1CCNCC1_5",
            "7": "O_0",
            "8": "C_0",
            "9": "c1ccncc1_0",
        },
        "block_smi": {
            "0": "c1ccccc1",
            "1": "CO",
            "2": "CO",
            "3": "C=O",
            "4": "C1CCNCC1",
            "5": "C1CCNCC1",
            "6": "C1CCNCC1",
            "7": "O",
            "8": "C",
            "9": "c1ccncc1",
        },
        "block_r": {
            "0": [0, 1, 2, 3, 4, 5],
            "1": [0, 1],
            "2": [1, 0, 0, 0],
            "3": [0],
            "4": [0, 3],
            "5": [1, 0, 3],
            "6": [3, 0],
            "7": [0],
            "8": [0],
            "9": [0, 1, 2, 4, 5],
        },
    }
    return debug_blocks


@pytest.fixture
def blocks_file(debug_blocks):
    with tempfile.TemporaryDirectory() as tmp_dir:

        blocks_file = Path(tmp_dir).joinpath("blocks_debug.json")
        with open(blocks_file, "w") as f:
            json.dump(debug_blocks, f)

        yield str(blocks_file)


def test_smoke_mol_mdp(blocks_file):
    """
    A simple smoke test that MolMDP can be instantiated.
    """
    random_seed = 333
    np.random.seed(random_seed)
    number_of_blocks = 5

    mdp = MolMDP(blocks_file=blocks_file)
    mdp.reset()
    mdp.random_walk(number_of_blocks)


