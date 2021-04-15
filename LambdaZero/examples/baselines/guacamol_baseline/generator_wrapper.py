from ray import tune
from pathlib import Path

from guacamol.scoring_function import ArithmeticMeanScoringFunction
from LambdaZero.examples.baselines.guacamol_baseline.smiles_lstm_hc.rnn_utils import load_rnn_model

from LambdaZero.examples.baselines.guacamol_baseline.smiles_ga.goal_directed_generation import *
from LambdaZero.examples.baselines.guacamol_baseline.smiles_lstm_hc.rnn_generator import *

Molecule = namedtuple('Molecule', ['score', 'smiles', 'genes'])
class ChemGEGenerator_wrapper(tune.Trainable):
    # rewrite of the generate optimized molecule as tune.Trainable

    def _setup(self, config):
        self.chemge = ChemGEGenerator(**config['method_config'])
        self.number_molecules = config["number_molecules"]
        self.starting_population = config["starting_population"]
        self.scoring_function = config["evaluator"]
        self.scoring_function = ArithmeticMeanScoringFunction([self.scoring_function])

        if self.number_molecules > self.chemge.population_size:
            self.chemge.population_size = self.number_molecules
            print(f'Benchmark requested more molecules than expected: new population is {self.number_molecules}')

        # fetch initial population?
        if self.starting_population is None:
            print('selecting initial population...')
            init_size = self.chemge.population_size + self.chemge.n_mutations
            all_smiles = copy.deepcopy(self.chemge.all_smiles)
            if self.chemge.random_start:
                self.starting_population = np.random.choice(all_smiles, init_size)
            else:
                self.starting_population = self.chemge.top_k(all_smiles, self.scoring_function, init_size)

        # The smiles GA cannot deal with '%' in SMILES strings (used for two-digit ring numbers).
        self.starting_population = [smiles for smiles in self.starting_population if '%' not in smiles]

        # calculate initial genes
        self.initial_genes = [cfg_to_gene(cfg_util.encode(s), max_len=self.chemge.gene_size)
                         for s in self.starting_population]

        # score initial population
        self.initial_scores = self.scoring_function.score_list(self.starting_population)
        self.population = [Molecule(*m) for m in zip(self.initial_scores, self.starting_population, self.initial_genes)]
        self.population = sorted(self.population, key=lambda x: x.score, reverse=True)[:self.chemge.population_size]
        self.population_scores = [p.score for p in self.population]

    def step(self):

        # old_scores = self.population_scores
        # select random genes
        all_genes = [molecule.genes for molecule in self.population]
        choice_indices = np.random.choice(len(all_genes), self.chemge.n_mutations, replace=True)
        genes_to_mutate = [all_genes[i] for i in choice_indices]

        # evolve genes
        joblist = (delayed(mutate)(g, self.scoring_function) for g in genes_to_mutate)
        new_population = self.chemge.pool(joblist)

        # join and dedup
        self.population += new_population
        self.population = deduplicate(self.population)

        # survival of the fittest
        self.population = sorted(self.population, key=lambda x: x.score, reverse=True)[:self.chemge.population_size]

        self.population_scores = [p.score for p in self.population]


        return {'max': np.max(self.population_scores), 'avg': np.mean(self.population_scores), 'min': np.min(self.population_scores), 'std': np.std(self.population_scores)}


class SmilesRnnDirectedGenerator_wrapper(tune.Trainable):
    def _setup(self, config):
        self.pretrained_model_path = config["method_config"]["pretrained_model_path"]
        self.n_epochs = config["method_config"]["n_epochs"]
        self.mols_to_sample = config["method_config"]["mols_to_sample"]
        self.keep_top = config["method_config"]["keep_top"]
        self.optimize_batch_size = config["method_config"]["optimize_batch_size"]
        self.optimize_n_epochs = config["method_config"]["optimize_n_epochs"]
        self.pretrain_n_epochs = 0
        self.max_len = config["method_config"]["max_len"]
        self.number_final_samples = config["method_config"]["number_final_samples"]
        self.sample_final_model_only = config["method_config"]["sample_final_model_only"]
        self.random_start = config["method_config"]["random_start"]
        self.smi_file = config["method_config"]["smi_file"]
        self.pool = joblib.Parallel(n_jobs=config["method_config"]["n_jobs"])


        self.number_molecules = config["number_molecules"]
        self.starting_population = config["starting_population"]
        self.scoring_function = config["evaluator"]

        # fetch initial population?
        if self.starting_population is None:
            print('selecting initial population...')
            if self.random_start:
                self.starting_population = []
            else:
                all_smiles = self.load_smiles_from_file(self.smi_file)
                self.starting_population = self.top_k(all_smiles, self.scoring_function, self.mols_to_sample)

        cuda_available = torch.cuda.is_available()
        self.device = "cuda" if cuda_available else "cpu"
        model_def = Path(self.pretrained_model_path).with_suffix('.json')

        self.model = load_rnn_model(model_def, self.pretrained_model_path, self.device, copy_to_cpu=True)

        self.rnn = SmilesRnnMoleculeGenerator(model=self.model,
                                               max_len=self.max_len,
                                               device=self.device)

        self.int_results = self.rnn.pretrain_on_initial_population(self.scoring_function, self.starting_population,
                                                          pretrain_epochs=self.pretrain_n_epochs)

        self.results: List[OptResult] = []
        self.seen: Set[str] = set()

        for k in self.int_results:
            if k.smiles not in self.seen:
                self.results.append(k)
                self.seen.add(k.smiles)

    def load_smiles_from_file(self, smi_file):
        with open(smi_file) as f:
            return self.pool(delayed(canonicalize)(s.strip()) for s in f)

    def top_k(self, smiles, scoring_function, k):
        joblist = (delayed(scoring_function.score)(s) for s in smiles)
        scores = self.pool(joblist)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]


    def step(self):
        samples = self.rnn.sampler.sample(self.rnn.model, self.mols_to_sample, max_seq_len=self.rnn.max_len)

        canonicalized_samples = set(canonicalize_list(samples, include_stereocenters=True))
        payload = list(canonicalized_samples.difference(self.seen))
        payload.sort()  # necessary for reproducibility between different runs

        self.seen.update(canonicalized_samples)

        scores = self.scoring_function.score_list(payload)
        int_results = [OptResult(smiles=smiles, score=score) for smiles, score in zip(payload, scores)]

        self.results.extend(sorted(int_results, reverse=True)[0:self.keep_top])
        self.results.sort(reverse=True)
        subset = [i.smiles for i in self.results][0:self.keep_top]

        np.random.shuffle(subset)

        sub_train = subset[0:int(3 * len(subset) / 4)]
        sub_test = subset[int(3 * len(subset) / 4):]

        train_seqs, _ = load_smiles_from_list(sub_train, max_len=self.rnn.max_len)
        valid_seqs, _ = load_smiles_from_list(sub_test, max_len=self.rnn.max_len)

        train_set = get_tensor_dataset(train_seqs)
        valid_set = get_tensor_dataset(valid_seqs)

        opt_batch_size = min(len(sub_train), self.optimize_batch_size)

        print_every = int(len(sub_train) / opt_batch_size)

        if self.optimize_n_epochs > 0:
            self.rnn.trainer.fit(train_set, valid_set,
                             n_epochs=self.optimize_n_epochs,
                             batch_size=opt_batch_size,
                             print_every=print_every,
                             valid_every=print_every)


        return {'Top score': self.results[0].score, 'Top smiles': self.results[0].smiles,
                'Second score': self.results[1].score, 'Second smiles': self.results[1].smiles}
