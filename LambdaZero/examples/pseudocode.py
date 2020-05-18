
class Querier:
    """
    stores an oracle and the queried points from it.
    """

    def __init__(self, oracle_calling_function):
        ...

    @property
    def unseen_points(self) -> set:
        """
        Method only exists when you know all the points you can query.
        """
        ...

    @property
    def seen_points_and_properties(self):
        ...

    def query_properties(points_to_query: list) -> list:

        # return properties
        ...

class BOModel:
    """
    Would be subclassed by BayesianRidgeRegression, Gaussian process etc...
    """

    def __init__(self, feature_extractor):
        """
        :param feature_extractor: This could be MPNN which converts molecule into
        embeddings or some other nn.Module. (or initially just a Rdkit fingerprint function.)
        Basically something which goes from a molecule to an embedding which gets fed into
         this model.
        """

    def fit(self, x_train, y_train):
        """
        This fits the Bayesian model but does not fit the feature extractor
        """
        ...

    def update(self, x_train, y_train):
        """ Some models allow you to update them more cheaply than retraining with all
        points (eg low rank update for Bayesian regression.) Hence you have an update
        function too.
        """
        ...

    def predict_mean_and_variance(self, x_test, full_cov=False):
        ...

    def loss_on_batch(self, x_train, y_train):
        """ This is useful if want to get loss all the way back eg to train the feature
        extractor too.
        """
        ...

class Acquirer:
    def __init__(self, available_points, model: BOModel, acquistion_function):
        """ :param aquistion_function: The reason why this is a function is because you may want to
        change it as you gather points (ie start random and then switch to batch BALD etc...)
        """
        ...

    def update_available_points(self, seen_points):
        ...

    def get_batch_points_to_query(self, batch_size):
        ...

# for num_batches:
#     points = Acquirer.get_batch_points_to_query()
#     points  = Querier.query_properties(points)
#     Acquirer.update_available_points(points)








# NOTE: All objects whose types are postfixed with Type are enums describing
#       classes available for use

def _getConfig() -> Config:
    argParser = ArgParser()
    argParser.parse()
    return Config(argParser)



def _getTrainable(dataSet: Type[DataSet], config: Config) -> Type[Trainable]:
    trainableBuilder = ObjectFactory.build(
        BaseTrainableBuilder,
        TrainableType[config.getSetting('TrainableType')],
        dataSet=dataSet,
        config=config
    )

    return trainableBuilder.build()


def main() -> int:
    config: Config = _getConfig()
    # In decagon case this becomes a DecagonDataSet
    dataSet: Type[DataSet] = DataSetBuilder.build(config)

    activeLearner: Type[BaseActiveLearner] = ObjectFactory.build(
        BaseActiveLearner,
        ActiveLearnerType[config.getSetting('ActiveLearnerType')],
        config=config
    )

    iterResults: Type[IterationResults] = None
    while activeLearner.hasUpdate(dataSet, iterResults):
        dataSet = activeLearner.getUpdate(dataSet, iterResults)

        trainable: Type[Trainable] = _getTrainable(dataSet, config)
        trainer: Type[BaseTrainer] = ObjectFactory.build(
            BaseTrainer,
            TrainerType[config.getSetting('TrainerType')],
            trainable=trainable,
            config=config
        )

        trainer.train()

        # Open question whether this should come from Trainer or Trainable
        iterResults = trainable.getIterationResults()

    return 0


###################################################
#       (For reference) Classes used above        #
###################################################

class DataSet:
    def __init__(
        self,
        adjacencyMatrices: AdjacencyMatrices,
        nodeFeatures: NodeFeatures
    ) -> None:
        self.adjacencyMatrices: AdjacencyMatrices = adjacencyMatrices
        self.nodeFeatures: NodeFeatures = nodeFeatures

class Trainable:
    '''
    This class is a DTO containing the objects necessary in order to
    correctly train a model.
    '''

    def __init__(self, dataSetIterator, optimizer, model):
        self.dataSetIterator = dataSetIterator
        self.optimizer = optimizer
        self.model = model

    # In the case of decagon (and others), this could be extended to return
    # the last layer of the network, the embeddings, train/val/test loss, etc.
    def getIterationResults(self) -> Type[IterationResults]:
        return IterationResults()

class BaseActiveLearner(BaseFactorizableClass, activeLearnerType=None, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config: Config) -> None:
        pass

    @abstractmethod
    def hasUpdate(
        self,
        dataSet: DataSet,
        iterResults: IterationResults
    ) -> bool:
        pass

    @abstractmethod
    def getUpdate(
        self,
        dataSet: DataSet,
        iterResults: IterationResults
    ) -> DataSet:
        pass

class IterationResults:
    '''
    (Empty) base class to be derived upon.  Will hold results from training
    that need to be used by the active learner in determining its next set
    of requested annotations.
    '''
    pass

class BaseTrainer(BaseFactorizableClass, trainerType=None, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, trainable: Trainable, config: Config) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass


# NOTE: Currently model evaluation is done by the DecagonLogger
#       (a subclass of BaseLogger).  The log method of the DecagonLogger
#       is called by the DecagonTrainer on specified iterations.  For
#       reference, DecagonLogger is provided below.
LOG_FILE_FORMAT = 'decagon_iteration_results_%d.csv'
PERC_IDX = LOG_FILE_FORMAT.find('%')
DOT_IDX  = LOG_FILE_FORMAT.find('.')

def _closeFile(f: _io.TextIOWrapper) -> None:
    f.close()

class BaseLogger(BaseFactorizableClass, metaclass=ABCMeta):
    def __init__(self, config: Config) -> None:
        self.numIterationsPerLog: int = int(config.getSetting('NumIterationsPerLog'))
        self.numIterationsDone: int = 0

    def incrementIterations(self):
        self.numIterationsDone += 1

    @property
    def shouldLog(self):
        return (self.numIterationsDone % self.numIterationsPerLog) == 0

    @abstractmethod
    def log(self, iterationResults: Type[TrainingIterationResults]) -> None:
        pass

class DecagonLogger(BaseLogger):
    '''
    Note that this class is not thread-safe
    '''
    def __init__(
        self,
        session: tf.Session,
        trainable: DecagonTrainable,
        checkpointer: TensorflowCheckpointer,
        config: Config
    ) -> None:
        super().__init__(config)

        self.trainResultLogFile: _io.TextIOWrapper = self._getTrainResultFile(config)
        self.trainResultWriter: DictWriter = self._getDictWriter()
        self.trainResultWriter.writeheader()

        self.checkpointer: TensorflowCheckpointer = checkpointer

        self.trainable: DecagonTrainable = trainable
        self.accuracyEvaluator: DecagonAccuracyEvaluator = DecagonAccuracyEvaluator(
            trainable.optimizer,
            trainable.placeholders,
            config
        )

        atexit.register(_closeFile, f=self.trainResultLogFile)

    def _getTrainResultFile(self, config: Config) -> _io.TextIOWrapper:
        return open(self._getTrainResultFileName(config), 'w')

    def _getTrainResultFileName(self, config: Config) -> str:
        baseDir = config.getSetting('TrainIterationResultDir')
        existingIndices = [
            self._getFnameIdx(f)
            for f in os.listdir(baseDir)
            if self._isValidFname(baseDir, f)
        ]

        thisFileIdx = 0
        if len(existingIndices) > 0:
            thisFileIdx = max(thisFileIdx)

        return LOG_FILE_FORMAT % thisFileIdx

    def _getFnameIdx(self, fname: str) -> int:
        stripPre = fname.lstrip(LOG_FILE_FORMAT[:PERC_IDX])
        stripPost = stripPre.rstrip(LOG_FILE_FORMAT[DOT_IDX:])

        return int(stripPost)

    def _isValidFname(self, baseDir: str, fname: str) -> bool:
        isFile = os.isfile(baseDir, fname)
        isGoodPrefix = fname[:PERC_IDX] == LOG_FILE_FORMAT[:PERC_IDX]
        isGoodPostfix = fname[DOT_IDX:] = LOG_FILE_FORMAT[DOT_IDX:]

        return isFile and isGoodPrefix and isGoodPostfix

    def _getDictWriter(self) -> csv.DictWriter:
        fieldnames = [
            'IterationNum',
            'Loss',
            'Latency',
            'EdgeType',
            'AUROC',
            'AUPRC',
            'APK'
        ]

        return csv.DictWriter(self.trainResultLogFile, fieldnames=fieldnames)

    @property
    def shouldLog(self):
        return super().shouldLog or self.checkpointer.shouldCheckpoint

    def incrementIterations(self) -> None:
        super().incrementIterations()
        self.checkpointer.incrementIterations()

    def log(
        self,
        feedDict: Dict,
        iterationResults: DecagonTrainingIterationResults
    ) -> None:
        if super().shouldLog:
            self._logInternal(feedDict, iterationResults)

        if self.checkpointer.shouldCheckpoint:
            self.checkpointer.save()

        return

    # Force log to filesystem and stdout at epoch end
    def logEpochEnd(
        self,
        feedDict: Dict,
        iterationResults: DecagonTrainingIterationResults
    ) -> None:
        self._logInternal(feedDict, iterationResults)
        self.checkpointer.save()

    def _logInternal(
        self,
        feedDict: Dict,
        iterationResults: DecagonTrainingIterationResults
    ) -> None:
        accuracyScores = self._computeAccuracyScores(feedDict)

        iterRowDict = self._getCsvRowDict(iterationResults, accuracyScores)
        iterString  = self._getString(iterationResults, accuracyScores)

        self.trainResultWriter.writerow(rowDict)
        print(iterString)

        return

    def _computeAccuracyScores(self, feedDict: Dict) -> AccuracyScores:
        iterator = self.trainable.dataSetIterator

        return self.accuracyEvaluator.evaluate(
            feedDict,
            iterator.idx2edge_type[iterator.current_edge_type_idx],
            iterator.val_edges,
            iterator.val_edges_false
        )

    def _getCsvRowDict(
        self,
        iterationResults: DecagonTrainingIterationResults,
        accuracyScores: AccuracyScores
    ) -> Dict:
        return {
            'IterationNum': self.numIterationsDone,
            'Loss': iterationResults.iterationLoss,
            'Latency': iterationResults.iterationLatency,
            'EdgeType': iterationRresults.iterationEdgeType,
            'AUROC': accuracyScores.auroc,
            'AUPRC': accuracyScores.auprc,
            'APK': accuracyScores.apk,
        }

    def _getString(
        self,
        iterationResults: DecagonTrainingIterationResults,
        accuracyScores: AccuracyScores
    ) -> str:
        '''
            IterationNum: %d
            Loss: %f
            Latency: %f
            EdgeType: %s
            AUROC: %f
            AUPRC: %f
            APK: %f



        ''' % (
            self.numIterationsDone,
            iterationResults.iterationLoss,
            iterationResults.iterationLatency,
            iterationResults.iterationEdgeType,
            accuracyScores.auroc,
            accuracyScores.auprc,
            accuracyScores.apk,
        )
