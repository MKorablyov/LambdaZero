import ray
import numpy as np
from LambdaZero.environments import MolMDP
from LambdaZero.examples.env3d.dataset.data_generation import get_data_row
from LambdaZero.examples.env3d.dataset.io_utilities import process_row_for_writing_to_feather


@ray.remote
class DataRowGenerator:

    def __init__(self, blocks_file: str, number_of_parent_blocks: int,
                 num_conf: int, max_iters: int, random_seed: int):

        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.num_conf = num_conf
        self.max_iters = max_iters
        self.number_of_parent_blocks = number_of_parent_blocks
        self.reference_molMDP = MolMDP(blocks_file=blocks_file)

    def generate_row(self):
        self.reference_molMDP.reset()
        self.reference_molMDP.random_walk(self.number_of_parent_blocks)
        number_of_stems = len(self.reference_molMDP.molecule.stems)

        if number_of_stems < 1:
            raise ValueError("no stems! Cannot proceed")

        attachment_stem_idx = np.random.choice(number_of_stems)

        row = get_data_row(
            self.reference_molMDP, attachment_stem_idx, self.num_conf,
            self.max_iters, self.random_seed
            )

        byte_row = process_row_for_writing_to_feather(row)

        return byte_row

