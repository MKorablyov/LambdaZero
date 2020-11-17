import numpy as np

def random_split(num_samples, probs):
    """ Split dataset with the number of samples according to probs
    :param num_samples: number of samples
    :param probs: sample probability for every group
    :return:
    """
    splits = [[] for _ in range(len(probs))]

    for i in range(num_samples):
        choice = np.random.choice(np.arange(len(probs)),1, p=probs)[0]
        splits[choice].append(i)
    splits = [np.asarray(s) for s in splits]
    return splits