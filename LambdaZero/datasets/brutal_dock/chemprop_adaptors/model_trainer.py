from LambdaZero.datasets.brutal_dock.model_trainer import AbstractModelTrainer


class ChempropModelTrainer(AbstractModelTrainer):
    target_name = 'gridscore'
    features_name = 'mol_graph'

    def _get_target_from_batch(self, batch):
        target = batch[self.target_name]
        target = target.to(self.device)
        return target

    def _apply_model_to_batch(self, batch, model):
        model.to(self.device)
        features = batch[self.features_name]
        y_hat = model.forward(features)
        return y_hat

    def _get_size_of_batch(self, batch):
        return len(batch[self.target_name])
