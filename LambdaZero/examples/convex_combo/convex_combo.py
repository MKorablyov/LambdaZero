import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns


class ConvexComboReward(torch.nn.Module):
    def __init__(self, embeds_path, support_size, num_feat):
        super(ConvexComboReward, self).__init__()
        self.kernel = torch.nn.Sequential(
            torch.nn.Linear(num_feat, num_feat, bias=False),
            torch.nn.ELU(),
            torch.nn.Linear(num_feat, 1, bias=False)
        )
        # todo: support size

        self.base_embeds, self.base_rewards = torch.load(embeds_path)

    def reset(self):
        pass

    def __call__(self, embeds):
        # todo: call should accept molecule
        # distances to the basis set
        dist = (embeds[:, None, :] - self.base_embeds[None, :, :])
        dist = torch.abs(self.kernel(dist)[:, :, 0])
        # distance weights
        eps = 1e-4
        dist_w = 1./(dist + eps)
        dist_w = dist_w / dist_w.sum(1)[:, None]
        rewards = (dist_w * self.base_rewards[None,:]).sum(1)
        return rewards

# todo: I could also learn individual distance function for each support set molecule
# todo: load and train kernels
class cfg:
    support_size = 125
    num_feat = 128
    embeds_path = "/home/maksym/Datasets/brutal_dock/seh/embeds/Zinc20_docked_neg_randperm_3k.pth"


base_embeds = torch.rand([5000, cfg.num_feat])
base_rewards = torch.rand([5000])
embeds = torch.rand([1000, 128])
torch.save([base_embeds, base_rewards],cfg.embeds_path)
torch.load(cfg.embeds_path)
comboReward = ConvexComboReward(cfg.embeds_path, cfg.support_size, cfg.num_feat)

rewards = comboReward(embeds)




# plot distribution of rewards
#sns.distplot(rewards.detach().numpy())
#plt.show()

# plot energy correlation
base_rewards = comboReward(comboReward.base_embeds)
plt.scatter(comboReward.base_rewards.detach().numpy(), base_rewards.detach().numpy())
plt.show()