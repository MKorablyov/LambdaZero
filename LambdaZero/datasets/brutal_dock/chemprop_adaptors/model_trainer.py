from LambdaZero.datasets.brutal_dock.model_trainer import AbstractModelTrainer


class ChempropModelTrainer(AbstractModelTrainer):
    """
    Trainer tailored to the chemprop model. Note that
    chemprop handles its own device, so this class won't
    use .to(device).
    """
    target_name = 'gridscore'
    features_name = 'mol_graph'

    def _get_target_from_batch(self, batch):
        return batch[self.target_name]

    def _apply_model_to_batch(self, batch, model):
        y_hat = model.forward(batch[self.features_name])
        return y_hat

    def _get_size_of_batch(self, batch):
        return len(batch[self.target_name])