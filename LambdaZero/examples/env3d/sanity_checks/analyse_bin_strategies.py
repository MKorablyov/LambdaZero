"""
The goal of this script is to consider different approaches to classifying a binned angle.
This is tricky because:
    - angles are 2pi periodic
    - ordinality matters; bins that are "close" to correct should have smaller loss than bins that are far

It's trickier than I thought. This is not useful work. Keep it around for a while, but it will probably be deleted.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from LambdaZero.models import nn

softmax = nn.Softmax(dim=1)

number_of_bins = 60
batch_size = number_of_bins

if __name__ == '__main__':
    np.random.seed(1231)

    binned_angles = 2*np.pi*torch.linspace(start=0., end=(1.-1./number_of_bins), steps=number_of_bins)
    basis_vectors = torch.stack([torch.cos(binned_angles), torch.sin(binned_angles)], dim=1)

    actual_bin_index = np.random.randint(0, number_of_bins)
    actual_angle = binned_angles[actual_bin_index]
    actual_vector = basis_vectors[actual_bin_index]

    # create a "batch" of actual labels
    actual_labels = torch.tensor(actual_bin_index).repeat(batch_size, 1)

    label_indices = torch.arange(number_of_bins)

    mask = torch.abs(torch.arange(number_of_bins).unsqueeze(0).repeat(batch_size, 1) - actual_labels)


    y = 100*torch.eye(batch_size, number_of_bins)
    s = softmax(y)

    delta_v = torch.mm(s, basis_vectors) - actual_vector.repeat(batch_size).reshape(batch_size, 2)

    distances_squared = torch.sum(delta_v**2, dim=1)
    distances = torch.sqrt(distances_squared)

    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(111)
    ax.grid(True)

    ax.plot(binned_angles, distances_squared, 'bo-', label='distances squared')

    ax.vlines(actual_angle, ymin=-0.5, ymax=4., color='red', label='actual angle')

    ax.legend(loc=0)

    plt.show()


