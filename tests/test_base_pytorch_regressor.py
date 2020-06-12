from LambdaZero.trainable.base_pytorch_regressor import BasePytorchRegressor


def test_combine_scores():
    train_scores_dict = {'loss': 1, 'other_loss': 2}
    validation_scores_dict = {'loss': 3, 'other_loss': 4}

    expected_scores = {'train_loss': 1, 'train_other_loss': 2,
                       'validation_loss': 3, 'validation_other_loss': 4}

    computed_scores = BasePytorchRegressor._combine_scores(train_scores_dict, validation_scores_dict)

    assert computed_scores == expected_scores
