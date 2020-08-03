import logging
import time
from pathlib import Path

import ray
import numpy as np

from LambdaZero.environments import MolMDP
from LambdaZero.examples.env3d.dataset.data_generation import (
    get_data_row,
    get_blocks_embedding_energies,
)
from LambdaZero.examples.env3d.dataset.io_utilities import (
    process_row_for_writing_to_feather,
)


@ray.remote
class DataRowGenerator:
    """
    This class drives the generation of the dataset, one row at a time, under the control of ray.
    """

    def __init__(
        self,
        blocks_file: str,
        number_of_parent_blocks: int,
        num_conf: int,
        max_iters: int,
        random_seed: int,
        logging_directory: Path,
    ):
        """
        Initialize the data generator.

        Args:
            blocks_file (str):  path to the file containing the vocabulary of blocks
            number_of_parent_blocks (int): number of blocks to use to build the parent molecule
            num_conf (int): number of conformer embeddings rdkit will try to converge the geometry
            max_iters (int): number of iterations rdkit will try to converge the geometry, for a given conformer
            random_seed (int): random seed used to generate random parents
            logging_directory (Path): path to the directory where logging files will be written.
        """

        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.num_conf = num_conf
        self.max_iters = max_iters
        self.number_of_parent_blocks = number_of_parent_blocks
        self.reference_molMDP = MolMDP(blocks_file=blocks_file)
        self.child_block_energies_dict = get_blocks_embedding_energies(blocks_file)

        self.logger = logging.getLogger(__name__)
        log_file_name = str(
            logging_directory.joinpath(f"info_ACTOR_seed_{random_seed}.log")
        )
        logging.basicConfig(filename=log_file_name,
                            format="%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() - %(message)s",
                            filemode="w",
                            level=logging.INFO)

        self.counter = 0

    def generate_row(self):
        self.logger.info(f" Starting row {self.counter}")
        t1 = time.time()
        self.reference_molMDP.reset()
        self.reference_molMDP.random_walk(self.number_of_parent_blocks)
        number_of_stems = len(self.reference_molMDP.molecule.stems)

        if number_of_stems < 1:
            self.logger.warning(f"no stems! Cannot proceed.")
            raise ValueError("no stems! Cannot proceed")

        attachment_stem_idx = np.random.choice(number_of_stems)

        row = get_data_row(
            self.reference_molMDP,
            attachment_stem_idx,
            self.child_block_energies_dict,
            self.num_conf,
            self.max_iters,
            self.random_seed,
        )

        byte_row = process_row_for_writing_to_feather(row)
        t2 = time.time()
        self.logger.info(
            f"     -> Finished row {self.counter}. Execution time: {t2-t1:5.1f} seconds"
        )
        self.counter += 1

        return byte_row
