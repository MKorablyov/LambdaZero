from LambdaZero.examples.baselines.Apollo1060.pipeliner_light.pipelines import ClassicPipe
from LambdaZero.examples.baselines.Apollo1060.pipeliner_light.smol import SMol
from guacamol.scoring_function import ArithmeticMeanScoringFunction, ScoringFunction, MoleculewiseScoringFunction, BatchScoringFunction

def apollo_scoringfunction(mol, pipe):
    smol = SMol(mol)  # standardization; takes both mol and str
    smol.featurize(pipe_dict[pipe].features)  # same intital features set before per-model selection
    return ccr5_pipe.predict_vector(smol.features_values)

    # scoring_fn = ArithmeticMeanScoringFunction([apo])
# ccr5_pipe = ClassicPipe.load('/project/def-perepich/pchliu/ML/LambdaZero/LambdaZero/examples/baselines/Apollo1060/Models/hiv_ccr5')
# int_pipe = ClassicPipe.load('/project/def-perepich/pchliu/ML/LambdaZero/LambdaZero/examples/baselines/Apollo1060/Models/hiv_int')
# rt_pipe = ClassicPipe.load('/project/def-perepich/pchliu/ML/LambdaZero/LambdaZero/examples/baselines/Apollo1060/Models/hiv_rt')
# ccr5_pipe = ClassicPipe.load('/Users/chenghaoliu/ML/LambdaZero/LambdaZero/examples/baselines/Apollo1060/Models/hiv_ccr5')
# int_pipe = ClassicPipe.load('/Users/chenghaoliu/ML/LambdaZero/LambdaZero/examples/baselines/Apollo1060/Models/hiv_int')
# rt_pipe = ClassicPipe.load('/Users/chenghaoliu/ML/LambdaZero/LambdaZero/examples/baselines/Apollo1060/Models/hiv_rt')
# pipe_dict = {'ccr5': ccr5_pipe, 'int': int_pipe, 'rt':rt_pipe}

class HIV_Scoringfunction:
    def __init__(self, apollo_pipe='ccr5') -> None:
        # self.apollo_pipe = ClassicPipe.load('/Users/chenghaoliu/ML/LambdaZero/LambdaZero/examples/baselines/Apollo1060/Models/hiv_'+apollo_pipe) #pipe_dict[apollo_pipe]
        self.apollo_pipe = ClassicPipe.load('/project/def-perepich/pchliu/ML/LambdaZero/LambdaZero/examples/baselines/Apollo1060/Models/hiv_' + apollo_pipe)  # pipe_dict[apollo_pipe]

    def __call__(self, smiles, *args, **kwargs):
        smol = SMol(smiles)  # standardization
        smol.featurize(self.apollo_pipe.features)  # same intital features set before per-model selection
        return self.apollo_pipe.predict_vector(smol.features_values)

class Apollo_Scoringfunction(BatchScoringFunction):

    def __init__(self, apollo_pipe='ccr5', score_modifier=None) -> None:
        self.apollo_pipe = pipe_dict[apollo_pipe]
        super().__init__(score_modifier=score_modifier)

    def raw_score_list(self, smiles_list):
        return [self._raw_score(i) for i in smiles_list]

    def _raw_score(self, smiles):
        smol = SMol(smiles)  # standardization
        smol.featurize(self.apollo_pipe.features)  # same intital features set before per-model selection
        return self.apollo_pipe.predict_vector(smol.features_values)

# smol = SMol('COC1=C(OC)C=C(CCN2C(C)=CC(C(COC3=CC(=CC=C3F)C#N)N3CCCC3CN(C3CC3)C(C)=O)=C2C)C=C1')  # standardization
# smol.featurize(ccr5_pipe.features)  # same intital features set before per-model selection
# print(ccr5_pipe.predict_vector(smol.features_values))