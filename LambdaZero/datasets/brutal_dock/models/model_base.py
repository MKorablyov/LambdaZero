import logging

import torch


class ModelBase(torch.nn.Module):
    """
    This base class for models implements useful factory methods so it is
    easy to create a model instance given a specific class and input parameters.
    """

    def __init__(self, **kargs):
        super(ModelBase, self).__init__()

    @classmethod
    def create_model_for_training(
            cls,
            model_instantiation_kwargs,
    ):
        """
        Factory method to create a new model for the purpose of training.
        :param model_instantiation_kwargs:
            dictionary containing all the specific parameters needed to instantiate model.
        :return:
            instance of the model.
        """
        model = cls(**model_instantiation_kwargs)

        return model

    @staticmethod
    def load_model_object_from_path(model_path: str):
        return torch.load(model_path, map_location=torch.device("cpu"))

    @classmethod
    def load_model_from_file(cls, model_path, model_instantiation_kwargs):
        """
        Factory method to instantiate model for evaluation.
        :param model_path: path to saved model on disk
        :param model_instantiation_kwargs: parameters needed for model instantiation
        :return: model object
        """

        logging.info(f"Instantiating model from checkoint file {model_path}")

        model_object = cls.load_model_object_from_path(model_path)
        model = cls(**model_instantiation_kwargs)
        model.load_state_dict(model_object)

        return model