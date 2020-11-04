Research Ideas
================

Here is a brief enumeration of research directions and results

John
----

`LambdaBO <https://www.notion.so/LambdaBO-6e78c9ec399844a6a05dc3491a3c6394>`__

Emmanuel
--------

Persistent Tree Search
~~~~~~~~~~~~~~~~~~~~~~

`Driver
code <https://github.com/MKorablyov/LambdaZero/blob/persistent_search_new_reward/LambdaZero/environments/persistent_search/persistent_tree.py>`__
\| `Training
Code <https://github.com/MKorablyov/LambdaZero/blob/persistent_search_new_reward/LambdaZero/examples/persistent_search/train_persistent_tree.py>`__

**Overview**: an always expanding search where nodes are sampled
according to some priority, to be explored by an RL agent. The RL agent
is only allowed to add molecule blocks to a given molecule, making this
search a DAG (in practice a tree).

**Algorithm**: \
  * Initialize tree with available blocks (optional, seed tree with topK molecules seen in previous runs) \
  * S = sample k nodes from tree \
  * Repeat: \
      * sample pi(a\|s), take step for each a,s -> s’ \
      * Train agent to predict reward and value (see below) \
      * Insert all s’ into tree \
      * For all (s,a,s’), if s’ is terminal, sample new node \
      * S = [S’ \| new samples] \
  * Every n iterations: \
      * Prune tree if it is larger than ``prune_at`` \
         * Find 25th percentile of rewards -> thresh \
         * Compute max(reward in subtree(node)) for all nodes \
         * Prune all nodes who’s max subtree reward is <= thresh \
      * Update for all nodes the value target for the RL actor, either: \
      * the max subtree/descendent reward \
      * the averaged montecarlo return of the subtree \
  * In parallel \
      * Compute PredDock reward for top nodes \
      * Compute SimDock reward for top nodes

* **Actor**: currently a GNN on the atom graph, outputs Q(s,a) where Q is
  trained to predict either the montecarlo return or the maximum reward of
  any descendent. The policy pi(a\|s) is a boltzmann policy with some
  temperature. 

* **Sampling**: nodes are sampling with probability p\_i /
  sum\_j p\_j, i.e. their priority over the sum of all priorities.
  Priority is currently computed as the value of the node (max subtree
  reward or montecarlo return). 

* **Reward**: The reward of a node is defined as the most precise reward computed so far. There are 3 levels:

1. A forward pass of the actor predicts the molecule’s reward (“free”
when node is expanded, 1ms/mol) 1. PredDockReward, which predicts Dock
sim with a pretrained MPNN (5ms/mol) 1. Dock sim reward, which predicts
the docking energy with some calculation (15s/mol)

Results
^^^^^^^

Loglinear reward hypothesis seems to hold (red is random search):
|image0|

in-RAM tree size does seem to matter, but perhaps not in the long run.
Interestingly a smaller buffer seems to be better early on. It may hurt
after 10M molecules but this takes a long time to reach (dotted lines
are top-1): |image1|

This algorithm also learns a nice pareto front of binding energy to
"discount" (i.e. synthesizability \* qed), color is time (red molecules
are the most recently found, blue the oldest): |image2|

TODO: ^ for boltzmann baseline

As time goes, this also finds molecules with higher energy but lower
discount, although this seems to converge (red is binned average):
|image3|

Persistent MCTS
~~~~~~~~~~~~~~~

I'm currently working on a persistent MCTS, I'm not yet sure how to best
parallelize it and handle the persitence part (which induces staleness
which presumably needs to be updated periodically).

Do we need to update priors? Check KL divergence

--------------

Maksym
------

--------------

Priyesh
-------

Walk Convolutions - `Project Plan <https://bit.ly/3hzgqLA>`__ Project
Details - `PPT <https://bit.ly/2ZDwCW8>`__

--------------

Moksh
-----

*Note: The current reward uses a MPNN trained to predict the docking
energy. However it is trained with only molecules with max energy of
about 3.2. So the reward (which is the dockscore \* (Synthesizability
discount \* QED discount) ) out at :math:`\sim2.9`, so most of the
methods max out around that value. This was also confirmed by running
Emmanuel's Persistent Search with and without a full docking simulation.
The runs without docking simulation maxed out at around 3 whereas it
reaches maximum rewards of around 4 with docking simulations (Emmanuel's
plot).*

.. figure:: https://i.imgur.com/QtkllxD.png
   :alt: 

Exploration Experiments
~~~~~~~~~~~~~~~~~~~~~~~

1. **Random Network Distillation**

`Relevant file in
code <https://github.com/MKorablyov/LambdaZero/blob/rnd_support/LambdaZero/models/ppo_rnd.py>`__
\| `Experiment
Folder <https://github.com/MKorablyov/LambdaZero/tree/rnd_support/LambdaZero/examples/PPO_RND>`__

`Random Network Distillation (RND) <https://arxiv.org/abs/1810.12894>`__
is a widely used exploration method. RND proposes adding an intrinsic
reward to encourage exploration of novel states.

.. raw:: latex

   \begin{equation}
       r^t = r_e^t + \alpha r_i^t
   \end{equation}

This intrinsic reward is computed with two randomly initialized
networks: predictor network :math:`f_\phi` and target network
:math:`f_{\phi*}`, which map the state to a :math:`d`-dimensional
embedding.

.. raw:: latex

   \begin{equation}
       r^t_i = {\lVert f_{\phi}(x_{t+1}) - f_{\phi*}(x_{t+1}) \rVert}^2
   \end{equation}

where :math:`x_{t+1}` is the normalized next state, and the reward
:math:`r_i^t` is normalized by maintaining a running mean and standard
deviation. The weights for the target network, :math:`\phi*` are fixed,
whereas the weights of the predictor :math:`\phi` are learned.
Essentially, the predictor tries to match the target networks (random)
output. So this reward will lower when the state is already visited.
This can be viewed as the quantification of uncertainty in prediction a
constant zero function. For learning with this intrinsic reward, we can
decompose the value function as :math:`V = V_i + \alpha V_e`, where
:math:`V_i` and :math:`V_e` represent the intrinsic and extrinsic value,
which can be two heads of a shared larger network and be learned with
different discount factors (:math:`\gamma_i` and :math:`\gamma_e`).
:math:`\alpha` is a hyperparameter used for assigning different
weightage to the intrinsic reward.

**Results** We incorporate this method with `Proximal Policy
Optimization <https://arxiv.org/abs/1707.06347>`__ and test it on the
``Block_Mol_Graph_v1`` environment in LambdaZero. We perform experiments
with different values of :math:`\alpha`. The embedding dimension for the
RND networks is fixed to 64.

|image4| |image5|

Results for the experiments with RND. The x-axis represents number of
steps (i.e. number of molecules, and the y-axis represents the
normalized reward). The green plot on the top is the PPO baseline. The
performance of RND with all values of :math:`\alpha` is worse than the
PPO baselines.

The performance of RND is significantly worse than the PPO baseline, in
terms of both, the mean as well max rewards. The mean reward is also
much smaller, indicating that the agent is unable to exploit the good
states that it encounters. This is also supported by looking at the
entropy of the policy. The entropy is high indicating the agent does not
reach a point where it starts to exploit, rather than keep exploring.

2. **Restart Buffer**

`Relevant File in
Code <https://github.com/MKorablyov/LambdaZero/blob/rnd_support/LambdaZero/environments/persistent_search/persistent_buffer.py>`__
\|
`Experiments <https://github.com/MKorablyov/LambdaZero/tree/rnd_support/LambdaZero/examples/PPO_RND/config.py>`__

We can also encourage exploration in the molecule space by starting new
episodes from previously encountered molecules. More specifically, a
molecule :math:`x_t` obtained at the end of an episode is added to a
buffer :math:`B`, if it's molecular fingerprint similarity is less than
a threshold :math:`\tau \in [0, 1]`, where :math:`\tau = 0.6` indicates
that two molecules are chemically different. And at the beginning of
each episode, we sample a molecule from the buffer as the starting
molecule. In principle this should encourage the agent to explore the
regions near molecules already discovered, which could be promising. The
size of the buffer :math:`|B|` is fixed, and on reaching that size, the
molecules with lower rewards are pruned.

**Results** We perform experiments with various values of :math:`\tau`
(0.5, 0.6, 0.7), and size of buffer fixed at 500,000. The experiments
are again performed with PPO, on a modified with the buffer. :math:`p`
is the probability of starting an episode from a random molecule instead
of a molecule from the buffer. :math:`pr` is the probabilty of adding a
molecule to the buffer, as an ablation for checking if the similarity
metric is useful (no threshold used in this case).

.. raw:: html

   <!-- ![](https://i.imgur.com/p5rQ4LT.png)
   ![](https://i.imgur.com/0hf21PN.png)
    -->

.. figure:: https://i.imgur.com/bMdWoxW.png
   :alt: 

Results for the experiments with the restart buffer. The x-axis
represents number of steps (i.e. number of molecules, and the y-axis
represents the normalized reward).

The buffer acts as an explicit constraint on the diversity of the
molecule regions that the agent explores. This results in poor
performance, since there is no mechanism to balance the exploration with
exploitation. So the agent keeps exploring without ever trying to
exploit the promising regions. Thus the performance suffers
considerably.

.. figure:: https://i.imgur.com/8yFyguZ.png
   :alt: 

|image6| |image7|

*Randomly add to buffer* |image8|

*Higher Thresholds* |image9|

Relevant Papers: https://arxiv.org/pdf/1811.11298.pdf
https://arxiv.org/pdf/1703.02660.pdf

**Things to try out** \* Higher threshold (allowing all mols to the
buffer) [Done] \* Start episodes with molecule from buffer or random
molecule with 0.5 probability for each [Done] \* Use the value instead
of reward to sort the buffer \* Add molecules to buffer randomly with
some probability p [done]

3. **Entropy Regularization**

`Experiments <https://github.com/MKorablyov/LambdaZero/blob/rnd_support/LambdaZero/examples/PPO/config.py>`__

Entropy Regularization has been shown to improve the optimization
landscape in Deep Reinforcement Learning. In settings with sparse
rewards, entropy regularization can help with exploration by encouraging
the agent to select different actions in similar situations. The entropy
of the current policy distribution is added as a regularizing term to
the loss.

.. raw:: latex

   \begin{equation}
       H(\pi(.|s_t)) = - \sum_{a \in A} \pi(a|s_t) \log (\pi(a|s_t))
   \end{equation}

We assign an entropy coefficient which controls the contribution of this
regularizing term to the overall loss function. This coefficient can
also be varied with time according to a schedule.

**Results** We perform experiments with different values and schedules
for the entropy coefficient(\ :math:`\beta`), with PPO as the base
algorithm on the ``Block_Mol_Graph_v1`` environment.

.. figure:: https://i.imgur.com/6GvQBbC.png
   :alt: 

.. figure:: https://i.imgur.com/64DPSDD.png
   :alt: 

Results for the experiments with entropy regularization. The x-axis
represents number of steps (i.e. number of molecules, and the y-axis
represents the normalized reward).

Entropy regularization, given the right coefficient, significantly
outperform the PPO baseline, in terms of both the best as well as
average performance. When set too high the agent is again, unable to
exploit promising regions. And if set too low, there will not be any
significant improvement over the baselines.

Environment Parameters
~~~~~~~~~~~~~~~~~~~~~~

`Experiments <https://github.com/MKorablyov/LambdaZero/blob/rnd_support/LambdaZero/examples/PPO/config.py>`__

Another important aspect that affects the performance is the environment
parameters. For instance, if ``num_steps``, which controls the maximum
number of steps in an episode, is set too low then the episode
trajectories will be shorter and the agent not be able to explore enough
in that particular region. We perform some experiments to determine the
parameters which perform well in general with PPO, assuming they would
generalize to other algorithms as well.

The parameters tuned are: ``num_steps``: maximum number of steps in an
episode, ``max_block``: maximum number of blocks allowed on a candidate
molecule, ``random_blocks``: number of random blocks at the start of an
episode.

.. figure:: https://i.imgur.com/OgAbL9D.png
   :alt: 

.. figure:: https://i.imgur.com/Mu06RXZ.png
   :alt: 

Results for the experiments with the environment parameters. The x-axis
represents number of steps (i.e. number of molecules, and the y-axis
represents the normalized reward).

Increasing ``random_blocks`` in principle should force the agent to
explore the block removal action more, since the initial randomly
assembled molecule will most likely not be good. However, this would
work only when the agent is allowed more steps in the episode. So
increasing both the number of steps and and random blocks (along with
max blocks) seems to perform well in terms of the top molecules
encountered. However, the mean performance suffers considerably since,
when starting with 4 random blocks the agent is unlikely to reach a good
molecule often. However, just increasing the number of steps and number
of blocks helps considerably and outperforms all other modifications.

AlphaZero
~~~~~~~~~

`Experiments <https://github.com/MKorablyov/LambdaZero/tree/rnd_support/LambdaZero/examples/AlphaZero>`__
`AlphaZero <https://science.sciencemag.org/content/362/6419/1140>`__
introduced a generalized policy iteration algorithm for learning to play
games such as chess using self-play. It uses a variant of the UCT
algorithm, *pUCT* with a value function and *prior* policy (not in the
traditional bayesian context) represented by neural nets. The action is
picked as follows.

.. raw:: latex

   \begin{equation}
       \arg \max_a [ Q(x, a) + \lambda_N \frac{\pi_\theta(a|x)}{\hat{\pi}(a|x)} ]
   \end{equation}

.. raw:: latex

   \begin{equation}
       \hat{\pi}(a|x) = \frac{1 + n(x, a)}{|A| + \sum_b n(x, b)}
   \end{equation}

.. raw:: latex

   \begin{equation}
       \lambda_N = c \frac{\sqrt{\sum_b n(x, b)}}{|A| + \sum_b n(x, b)}
   \end{equation}

where :math:`\pi_\theta` is the policy prior and :math:`Q` is
action-value function, and :math:`n(x,a)` is the visit count of action
:math:`a` at state :math:`x` in the MCTS. To use AlphaZero in the
LambdaZero setting, we use `ranked
rewards <https://arxiv.org/abs/1807.01672>`__.

It can be shown that AlphaZero approximates the solution of a
regularized policy optimization problem. Using this insight, `Monte
Carlo Tree Search as Regularized Policy
Optimization <https://arxiv.org/abs/2007.12509>`__ proposes using exact
solution of this regularized policy optimization problem. So instead of
using the empirical visit distribution (:math:`\hat{\pi}`), we use the
solution of the policy optimization (:math:`\bar{\pi}`) instead.

.. raw:: latex

   \begin{equation}
       \bar{\pi} = \arg \max_{y \in S} [Q(x, .)^T y - \lambda_N KL[\pi_\theta(.|x), y]]
   \end{equation}

where :math:`S` is an :math:`|A|`-dimensional simplex, and :math:`KL` is
the KL divergence. This can also be reformulated as follows:

.. raw:: latex

   \begin{equation}
       \bar{\pi} = \lambda_N \frac{\pi_\theta(.|x)}{\alpha - Q(s,.)}
   \end{equation}

where :math:`\alpha` is selected such that :math:`\bar{\pi}` is a valid
probability distribution. We act and search according to
:math:`\bar{\pi}` instead of :math:`\hat{\pi}` and Equation (4)
respectively. The prior policy is updated with :math:`\bar{\pi}`. Since
:math:`\bar{\pi}` depends on the Q values instead of the number of
visits, the policy can learn promising new actions without requiring a
large number of simulations. Thus, this formulation would help
considerably in scenarios where the simulation budget is low.

**Results** We compare AlphaZero with the proposed policy optimization
based modification(AlphaZero-PO) as well as a PPO baseline. We perform
experiments with different simulation budgets (100, 200, 800).

|image10| |image11| Results for the experiments with AlphaZero, with
number of simulations as 800. The x-axis represents number of steps
(i.e. number of molecules, and the y-axis represents the normalized
reward).

|image12| |image13| Results for the experiments with AlphaZero, with
number of simulations as 200. The x-axis represents number of steps
(i.e. number of molecules, and the y-axis represents the normalized
reward).

|image14| |image15| Results for the experiments with AlphaZero, with
number of simulations as 100. The x-axis represents number of steps
(i.e. number of molecules, and the y-axis represents the normalized
reward).

|image16| |image17|

|image18| |image19|

AlphaZero consistently outperforms PPO by a significant margin, even
when number of simulations is set to 100. The policy optimization based
modification also seems to improve the performance over AlphaZero. It
reaches the maximum reward much faster than the AlphaZero baseline.
However since the rewards max out, it is hard to quantify to what
extent.

|image20| |image21|

**Things to try out** \* Lower learning rates. [Done] \* Lower number of
simulations [Done]

LambdaBO (BayesOpt + RL)
~~~~~~~~~~~~~~~~~~~~~~~~

`Code <https://github.com/MKorablyov/LambdaZero/blob/bayesian_reward/LambdaZero/environments/reward.py#L387>`__
\|
`Experiments <https://github.com/MKorablyov/LambdaZero/tree/bayesian_reward/LambdaZero/examples/bayesian_models/rl>`__
One of the main aspects of LambdaZero is to learn to leverage multiple
oracles with different fidelities and computational cost to guide search
through the space of molecules. Bayesian Optimization literature has lot
of work in this problem setting. Most of these methods however do not
scale well to higher dimensional problems. The idea (proposed in John
and Miguel's writeup) is to use reinforcement learning as the inner loop
for collecting samples for the bayesian optimization. The RL agent will
in turn recieve a signal from the bayesian model as it's reward.

In the reinforcement learning setup, we have an MDP where the state is
the current molecule and the actions are . The reward :math:`R` is
defined as the value of a bayesian acquisition function defined on the
model with uncertainty (say UCB (:math:`R = \mu + \beta * \sigma`) where
:math:`\mu` is the mean and :math:`\sigma` is the deviation of the
reward prediction). Whenever the RL agent suggests a molecule to
evaluate at the end of the episode, it is added to a buffer.

Every :math:`n` molecules collected in the buffer, we acquire the batch
with the highest acquistion value and retrain the model with uncertainty
with this acquired batch. The RL agent then uses this updated model for
the reward.

.. figure:: https://i.imgur.com/fLQk8cB.png
   :alt: 

Results
^^^^^^^

.. figure:: https://i.imgur.com/FkCASGc.png
   :alt: 

Uncertainty Learning
~~~~~~~~~~~~~~~~~~~~

Link to notebook:
https://colab.research.google.com/drive/1M2\_Tq6m09ySHCffYshj1GLW7O24s6HxD?usp=sharing

Algorithm (Regression): *To be updated*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pretrain a Kernel Density Estimator :math:`d` on training data
:math:`x\sim D`. :math:`L_a` = Initialize() :math:`L_f` = Initialize()
:math:`L_e` = Initialize() Initialize :math:`D_{ood}` with OOD samples
(uniformly sampled) t = 0 **repeat** for batches (x, y) from :math:`D`:
\* sample :math:`(x_{ood}, y_{ood}) \sim D_{ood}` \* **if** t mod N == 0
then: \* y2 ~ P(Y \|x) \* data = ((x, y),(x, y2))) \* errors = (([x,
d(x)], e(f(y), y)),([x, d(x)], e(f(y), y2)), ([:math:`x_{ood}`,
d(\ :math:`x_{ood}`)], e(f(\ :math:`x_{ood}`), :math:`y_{ood}`))) \*
Update(\ :math:`L_a`,(x,(y-y2)^2 / 2)) \* **else** \* errors = (([x,
d(x)], e(f(x), y)), ([:math:`x_{ood}`, d(\ :math:`x_{ood}`)],
e(f(\ :math:`x_{ood}`), :math:`y_{ood}`))) \* data = ((x, y)) \* **end
if** \* Update(\ :math:`L_f`, data) \* Update(\ :math:`L_u`, errors) \*
retrain :math:`f` and :math:`e` on all data seen so far \* t +=1

YB: it would be good to show the pseudo-code with all the bells and
whistles (i.e. the extension with OOD data for :math:`\hat{u}` and
density estimator).

Extensions: 1. Train :math:`\hat{u}` with additional information, for
instance the density estimated by a density estimation model pretrained
on the training data.

YB: the density estimate seems too smooth (you should make sure to
cross-validate the kernel width).

3. Uniformly sample OOD points and use these to train :math:`\hat{u}` as
   well. How are the OOD points used for training? In each iteration, a
   batch of OOD samples is added to the training batch for
   :math:`\hat{u}`.

.. raw:: html

   <!-- YB: yes, althought it would be interesting to see how many OOD examples are needed for this to work well. Hopefully a small fraction (like 10% or 20% of each batch as OOD examples is sufficient, although the answer may depend on the problem and dimensionality).

   The training points are from the function $\sin 2 \pi x$ with gaussian noise($\mu=0$, $\sigma=0.1$). 

   **Without OOD Samples**:
   ![](https://i.imgur.com/QcYhunN.png)
   ![](https://i.imgur.com/buLde2I.png)


   After tuning KDE:
   ![](https://i.imgur.com/64dQsGM.png)

   ![](https://i.imgur.com/VJVztRz.png)

   ![](https://i.imgur.com/NHQru2m.png)

   ![](https://i.imgur.com/k6iKrWA.png)

   ![](https://i.imgur.com/jzN5lI2.png)


   Setting Frequency N=3:

   ![](https://i.imgur.com/lmgNead.png)

   ![](https://i.imgur.com/clCaUdz.png)

   <!-- ![](https://i.imgur.com/qSCkSkG.png) -->

   <!-- **With OOD Samples**
   ![](https://i.imgur.com/4Clfyhs.png)

   YB: The u_loss curve seems to have some trouble initially. Maybe learning rate too large initially. It looks like u still has some way to go before converging. -->

   <!-- ![](https://i.imgur.com/JfD2bGg.png) -->
   <!-- ![](https://i.imgur.com/ZhVcNBj.png) -->


   <!-- YB: Nice but still underestimates uncertainty in several places. What fraction of minibatches of $\hat{u}$ are OOD? [MJ: 50%]
   And the density estimator is probably too smooth. It would be good to show the OOD points used to train $\hat{u}$.[MJ: Done]

   After Tuning KDE:
   ![](https://i.imgur.com/D5YGsPf.png)
    -->

**Newer Results**: Previously due to an error the network was being
trained on the density of the estimator, but at test time the log
density was being used, leading to wrong estimates.

**Without OOD**:

|image22| |image23| |image24| |image25| |image26|

**With OOD** |image27| |image28| |image29| |image30| |image31|

**Switch to estimating e instead of u** Also randomly sample points as
N/2 OOD points |image32| |image33| |image34|

For comparision Squared Error |image35|

MSE: Our Approach: 0.016015647 MC Dropout: 0.22903986 Single Model
Uncertainties: 0.1424897

-  `Single-Model Uncertainties for Deep Learning [NeurIPS
   2019] <https://arxiv.org/pdf/1811.00908.pdf>`__ on the same data
   |image36|

-  `MC Dropout <https://arxiv.org/pdf/1506.02142.pdf>`__ |image37|

-  Gaussian Proccess |image38|

Comments: \* :math:`f` is not trained on the uniformly sampled examples
so they are OOD, thus we can measure the true out sample error for these
points. \* There is L2 regularization on :math:`\hat{a}`, otherwise it
sometimes just collpases to zero. \* As of now, the OOD samples are not
added to the retraining for :math:`\hat{u}` step, which would explain
why there is not much difference when they are added. Next thing to try
is add these points in retraining for :math:`\hat{u}`. \* Another thing
to note is the networks are extremely sensitive to the learning rate,
and require lot of careful tuning.

**Relevant Papers to look at** [*Very similar, uses loss prediction for
active learning*\ ] https://arxiv.org/pdf/1905.03677.pdf

http://www.columbia.edu/~jwp2128/Papers/LiuPaisleyetal2019.pdf
https://arxiv.org/pdf/1506.02142.pdf
https://arxiv.org/pdf/1811.00908.pdf

**Things to try**: \* log density to be used as feature \* Add OOD
samples to retraining step for :math:`\hat{u}` [Done] \* Maybe use other
models to estimate density, instead of KDE. \* YB: try larger kernel
(use cross-validation) for KDE. [Done] \* YB: show the OOD samples on
the figu re [Done] \* YB: make sure :math:`\hat{u}` is well trained
(which requires keeping the OOD samples in its training set, indeed).
Maybe play with learning rate schedule or adaptive learning rate method.
\* YB: compare with other baselines (in particular MC-dropout). [GP
Partly Done] \* YB: Active learning not a fixed set points, but on
environment? (Dropout paper) \* YB: Using our uncertainty estimator for
active learning

--------------

Sumana
------

--------------

Howard
------

--------------

Victor
------

Reward Predictor
~~~~~~~~~~~~~~~~

Instead of using full on docking to approximate how well our candidate
molecule binds to a target, it would be incredibly useful to have a
cheap approximator that could accurately predict binding energy.

Note: one metric used many times over the course of these experiments is
"regret". Over the course of feeding batches of data (batches of mols in
this case), there exists some ranking of the all the data seen so far by
some metric; for this experiment its the energy of the molecule and we
rank low to high. Whenever we feed data to the model, it produces a
prediction for all the mols in the batch, and from these predictions,
and all those seen before, we construct a second ranking where we rank
from lowest to highest; this time, the lowest being the real energy from
the mol corresponding to the lowest predicted energy. The difference
between these two rankings is regret, and we consider the difference of
medians, MAE, and MSE regret.

Backpropagated Loss
~~~~~~~~~~~~~~~~~~~

One thing to try first was in general, do we see a perfomance difference
between backpropagating L2-loss vs L1-loss? |image39| |image40|

Preliminarily it seems that for the task of learning our MPNN it doesn't
seem to make a noticable difference; however, for the purpose of fine
tuning the reward predictor, we still try backpropagating L2 and L1 loss
for each sampler scheme.

Non-uniform sampling of molecules for training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the attempt to focus on getting good accuracy for molecules with
lower energy, one attempt that we studied fairly extensively has been
how oversampling lower energy molecules affects preformance in terms of
regret.

To do this we changed our dataloaders from just iterating through the
dataset without replacement to sampling with replacement, where each
molecule was sampled with probability
:math:`p = (\frac{\text{mol energy}}{\text{sum of mol energies}})^n + \epsilon`
where n is a hyper-parameter determining how aggresively we oversample
high energy molecules.

Results
~~~~~~~

In order to show the runs clearly, here is first a high level analysis
of the regret curves for top 15 median difference for regret and top 50
median difference for regret.

.. figure:: https://i.imgur.com/HLMul4g.png
   :alt: 

It seems from many different sampling schemes, with powers of n ranging
between 1 to 5, and a run with power 20, there doesn't seem to be a
difference in the prediction of the 15th or 50th ranked energy on our
validation set.

Note: It was brought up at the last meeting that maybe we should be
measuring our preformance on the training set; however, since we are
expecting this model to transfer to any manner of molecule we might see
in our search, we think this might be more representative of the models
ability to distinguish high energy molecules.

Training without low energy tail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our training and validation sets are composed of uniformely distributed
(acording to their energy) molecules, except for a small "tail" towards
the low end of the energy distribution where there is a much small group
of very low energy molecules.

Because in reality we hope to see molecules with even lower energy than
what we have in our training/validation sets, we wanted to investigate
how well our model would generalize to the validation set with higher
energies included, if we didn't train with the tail included. Here are
the results:

.. figure:: https://i.imgur.com/sVEkzmR.png
   :alt: 

The sit of curves that converge in the top group were all trained with
no tail, and those below with. Its clear that our model overfits to the
training data, and doesn't generalize well to higher-energy molecules.
This is an issue that is known with our RL experiments but is reinforced
here.

**Things to try out** \* Try different MPNN architecture, especially
directed message passing neural networks (dimenet)

--------------

Joanna
------

1. Experiments Setup
^^^^^^^^^^^^^^^^^^^^

We begin this section by describing our task and the dataset we use. We
describe our base architecture before then going on to describe two ways
we convert it into a Bayesian model to enable the generation of
uncertainties. We then detail how we evaluate our models, before
finishing by describing our acquisition function (i.e. how we use our
model to decide on which molecules to obtain the data for next). The
next section goes onto evaluate these models.

Let :math:`G` be the space of molecule graphs. Our base model is a
Message Passing Neural Network (MPNN) :math:`f: G \rightarrow R^{64}`
and 2 layers MLP :math:`g: R^{64} \rightarrow R` with dropout on the
last layer only, as default. Let :math:`\theta=\{\theta_f, \theta_g\}`.
The objective function is :math:`min_\theta\sum_{x,y}||g(f(x)) - y||^2`
where :math:`x \in G` and :math:`y` is the ground truth. The datasets we
use are Zinc15\_2k and Zinc15\_260k. We will denote the number of
datapoints, indicated by the number after the underscore in the dataset
name (i.e. 2k in 'Zinc15\_2k'), by :math:`n` where
:math:`n \in \{2000, 260000\}`.

There are two ways to convert our base model to turn them into Bayesian
models to allow the calculation of uncertainties. Each will give a
posterior predictive distribution based on mean and variance to compute
uncertainty and we can use log-likelihood to evaluate how good these
values are. These are as follows: 1. MPNN + MC dropout: Use Gal and
Ghahramani [2016] to compute mean and variance. The log-likelihood to
evaluate uncertaintainties is

.. math::


   \log p\left(\mathbf{y}^{*} \mid \mathbf{x}^{*}, \mathbf{X}, \mathbf{Y}\right) \approx \operatorname{logsumexp}\left(-\frac{1}{2} \tau\left\|\mathbf{y}-\widehat{\mathbf{y}}_{t}\right\|^{2}\right)-\log T-\frac{1}{2} \log 2 \pi-\frac{1}{2} \log \tau^{-1}

where :math:`T` is the number of forward passes performed, and
:math:`\tau`, the precision, is set from the hyperparameter as
:math:`\tau=\frac{p l^{2}}{2 N \lambda}` where :math:`p` is :math:`1 -`
dropout probability, :math:`l` is the prior length-scale, :math:`N` is
the length of training set and :math:`\lambda` is the weight decay
hyperparameter. In our case, we set the values T = 20, p = 0.9 and
:math:`\lambda` = 1e-8 as default, and later do grid search to find the
optimal value for :math:`\lambda`.

2. Bayesian: Use Bayesian ridge regression to compute mean and variance.
   After the training is done, we get the embeddings
   :math:`E_{64 \times n}` before the last layer of the MLP and fit a
   Bayesian ridge regression :math:`h: E \rightarrow y` without using MC
   dropout. The log-likelihood to evaluate uncertaintainties is

   .. math::


      l(\hat\mu, \hat\sigma^{2}; y)=-\frac{n}{2} \ln (2 \pi\hat\sigma^{2})-\frac{1}{2 \hat\sigma^{2}} \sum_{j=1}^{n}\left(y_{j}-\hat\mu\right)^{2}.

| As well as log likelihood we also consider the following metrics: -
MAE (mean absolute error)
| - MSE (mean squared error) - Aq\_Top15: Median of sorted top 15 ground
truths from acquirer
| - Aq\_Top50: Median of sorted top 50 ground truths from acquirer
| - Top 15 regret: Let :math:`y^{gt} = \{y_{g(1)},..,y_{g(N)}\}` be the
top N energies ranked accordin to their ground truth values and and
:math:`y^{model} = \{y_{m(1)},..,y_{m(N)}\}` be the top N energies
ranked according to the predicted energy scores from the model. top 15
regret =
:math:`\text{median}\{y_{g(1)},..,y_{g(15)}\} - \text{median}\{y_{m(1)},..,y_{m(15)}\}`.
- Top 50 regret: similarly, top 50 regret =
:math:`\text{median}\{y_{g(1)},..,y_{g(50)}\} - \text{median}\{y_{m(1)},..,y_{m(50)}\}`.

Then we added an Upper Confidence Bound (UCB) acquisition function to
acquire data. Suppose :math:`D_t` is the dataset for training and
:math:`D_r` is the remaining dataset. Everytime we train a Bayesian
model on :math:`D_t` and apply it to :math:`D_r` to get the above
metrics. We use the mean :math:`\hat\mu` and variance
:math:`\hat\sigma^{2}` on :math:`D_r` from posterior predictive
distribution given by the Bayesian model to compute UCB scores such that
:math:`\text{UCB scores} = \hat\mu(x) + \kappa \hat\sigma^{2}(x)`. We
select points from :math:`D_r` based on the UCB scores and add them to
:math:`D_t` for training. Repeat this process.

2. Results
^^^^^^^^^^

2.1 Bayesian model on the embeddings without using UCB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The results of Bayesian ridge regression on the embeddings are shown in
Table 1. It is trained over 61 epochs on 2k dataset and 20 epochs on
260k dataset, and the metrics in Table 1 are recorded at the end of
testing. While the MSE of the test loss shows it overfits a bit, the
260k performs better than 2k according to the MSE, log-likelihood, top
15 and 50 regret on the test set.

Table 1. Bayesian models on the embeddings using 2k and 260k datasets
without using UCB

+-----------+--------+---------+-------------------------+-----------------+-----------------+
| Dataset   | MAE    | MSE     | MPNN+Bayesian log-lik   | top 15 regret   | top 50 regret   |
+===========+========+=========+=========================+=================+=================+
| 2k        | 2.43   | 10.37   | -0.67                   | 2.35            | 2.13            |
+-----------+--------+---------+-------------------------+-----------------+-----------------+
| 260k      | 1.38   | 3.11    | -0.12                   | 1.77            | 0.67            |
+-----------+--------+---------+-------------------------+-----------------+-----------------+

2.2 Tuning hyperparameter :math:`\lambda`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the MC-Dropout model variant, we want to find the regularization
term coefficient :math:`\lambda` which gives the highest log-likelihood.
Consider MPNN with three dropout cases: (1) dropout on the last layer
only (2) dropout on the weights of the entire network, including the
last layer (3) dropout on input and output data, weights, and the last
layer. Each of these different case will add varying levels of
uncertainty.

For each case, we did grid search on ten :math:`\lambda` values which
were equally partitioned from the range of 1e-4 to 1e-12; they are
{1e-4, 1.29e-5, 1.67e-6, 2.15e-7, 2.78e-8, 3.59e-9, 4.61e-10, 5.99e-11,
7.74e-12, 1e-12}. We included the best result of each metric after 61
epochs on 2k dataset and after 20 epochs on 260k dataset as well as the
corresponding :math:`\lambda` in Table 2. In the table, the optimal
lambda which gives the best log-likelihood is consistent on both
datasets where :math:`\lambda^*`\ =1e-9 on the 2k dataset and
:math:`\lambda^*` = 1e-11 on the 260k dataset. Moreover, these
:math:`\lambda^*` values give the global maximum on [1e-4, 1e-12]; the
likelihood increases as :math:`\lambda` decreases until the value
reaches the maxima at each :math:`\lambda^*`, after which, the
likelihood decreases. Then we did grid search within a contracted range
from 1e-8 to 1e-12 again. We got a consistent conclusion and a more
precise :math:`\lambda` that the dropout case (2) gives the highest
log-likelihood on the 2k dataset with $^\*$ = 6.16e-9, and (2)(3) both
give the highest log-likelihood with :math:`\lambda^*` = 5.99e-11 on the
260k dataset.

Comparing Table 2 with Table 1, we can see that on the 2k dataset, MC
dropout gives slightly higher log-likelihood in the dropout case (2); on
the 260k dataset, MC dropout gives much higher log-likelihood than
Bayesian model in all three dropout cases.

Table 2. Grid search on :math:`\lambda` ranges from 1e-4 to 1e-12 with
three different dropout cases

+-----------+-------------------------------+---------------------------------+----------------------------------+----------------------------------+---------------------------------------------------+-------------------------------------+
| Dataset   | Dropout                       | MAE                             | MSE                              | **log-lik**                      | top 15 regret                                     | top 50 regret                       |
+===========+===============================+=================================+==================================+==================================+===================================================+=====================================+
| 2k        | Last layer only               | 2.10 (:math:`\lambda`\ =e-9)    | 8.9 (:math:`\lambda`\ =e-9)      | -0.73 (:math:`\lambda`\ =e-9)    | 0 (:math:`\lambda`\ =e-9)                         | 7e-3 (:math:`\lambda`\ =e-9)        |
+-----------+-------------------------------+---------------------------------+----------------------------------+----------------------------------+---------------------------------------------------+-------------------------------------+
| 2k        | Last layer + weights          | 2.25 (:math:`\lambda`\ =e-5)    | 8.9 (:math:`\lambda`\ =e-12)     | -0.65 (:math:`\lambda`\ =e-9)    | 0 (:math:`\lambda`\ =e-5,e-6,e-7,e-8,e-11,e-12)   | 0.1 (:math:`\lambda` =e-7, e-11)    |
+-----------+-------------------------------+---------------------------------+----------------------------------+----------------------------------+---------------------------------------------------+-------------------------------------+
| 2k        | Last layer + weights + data   | 2.40 (:math:`\lambda`\ =e-6)    | 9.99 (:math:`\lambda`\ =e-10)    | -0.69 (:math:`\lambda`\ =e-9)    | 0 (:math:`\lambda`\ =e-8,e-9,e-11,e-12)           | 0.11 (:math:`\lambda`\ =e-12)       |
+-----------+-------------------------------+---------------------------------+----------------------------------+----------------------------------+---------------------------------------------------+-------------------------------------+
| 260k      | Last layer only               | 1.36 (:math:`\lambda`\ =e-6)    | 3.002 (:math:`\lambda`\ =e-6)    | -0.04 (:math:`\lambda`\ =e-11)   | 1.12 (:math:`\lambda`\ =e-9)                      | 1.42 (:math:`\lambda`\ =e-11)       |
+-----------+-------------------------------+---------------------------------+----------------------------------+----------------------------------+---------------------------------------------------+-------------------------------------+
| 260k      | Last layer + weights          | 1.34 (:math:`\lambda`\ =e-11)   | 2.92(\ :math:`\lambda`\ =e-10)   | -0.01 (:math:`\lambda`\ =e-11)   | 1.44 (:math:`\lambda`\ =e-12)                     | 0.92 (:math:`\lambda`\ =e-11)       |
+-----------+-------------------------------+---------------------------------+----------------------------------+----------------------------------+---------------------------------------------------+-------------------------------------+
| 260k      | Last layer + weights + data   | 1.34 (:math:`\lambda`\ =e-7)    | 2.90 (:math:`\lambda`\ =e-7)     | -0.01 (:math:`\lambda`\ =e-11)   | 1.38 (:math:`\lambda`\ =e-6)                      | 1.23 (:math:`\lambda`\ =e-5,e-12)   |
+-----------+-------------------------------+---------------------------------+----------------------------------+----------------------------------+---------------------------------------------------+-------------------------------------+

2.3 UCB random baseline
^^^^^^^^^^^^^^^^^^^^^^^

We added a UCB acquisition function to acquire data. First we acquired
data using UCB but with a large noise (:math:`\epsilon = 1000`) on the
large and small datasets and let it be the random baseline. We put the
results of UCT random baseline, Bayesian model, MPNN with default
:math:`\lambda` together for each experiment on 260k. The top 15 and top
50 regrets are shown in Figure 1. Note that as epochs increase, the
number of datapoints in :math:`|D_t|` increases and the number of
datapoints in :math:`|D_r|` decreases. Although we only ran the random
acquisition for 20 epochs, we can still see at the same point in time
using a Bayesian or MC dropout model would have given better results at
20 epochs.

|Top 15 and Top 50 regret of three different dropout cases on 260k
dataset| Figure 1: Top 15 and Top 50 regret of three different dropout
cases on 260k dataset

2.4 Tune :math:`\kappa`
^^^^^^^^^^^^^^^^^^^^^^^

To explore the exploration/exploitation tradeoff, we did grid search on
:math:`\kappa \in \{0.01, 0.1, 1, 10\}` over 20 epochs on the 260k
dataset twice and got inconsistent results, the inconsistency can be
ignored because the number of epochs at which we reach covergence is
very similar between experiments (Figure 2). In Figure 3, we compare the
results with random baseline together. Results also shows that all
kappas works similar to each other and random baseline performs much
worse. |Results from running grid search of kappa on 260k dataset twice|
Figure 2: Results from running grid search of kappa on 260k dataset
twice

|Results from random baseline and grid search of kappa on 260k dataset|
Figure 3: Results from random baseline and grid search of kappa on 260k
dataset

--------------

Samin
-----

Graph\_bottleneck
~~~~~~~~~~~~~~~~~

**MDP**

At timestep :math:`t`, state :math:`s_t \in S` is attribute (feature) of
the current node, action :math:`a_t \in A` is choosing the next node to
pass the message. Reward for passing message through the nodes is
computed using :math:`v_\phi(s,a)`. The objective function
:math:`v_\phi` want to maximize: :math:`J(\phi) =`

**REINFORCE in practice**

" sample :math:`{ \{ s_t,a_t ... s_T\} } \sim \pi_{\theta}(.)` Update
value approximation :math:`v_\phi` using
:math:`\phi \leftarrow \phi + \nabla_\phi J(\phi)` (need to define
:math:`J(\phi)` ) for each timestep :math:`t=\{0,1..T-1\}`:
----:math:`G_t = \sum_{k=t}^T\gamma^k r(s_k,a_k)`
:math:`\min \nabla_\theta J(\theta) = -\sum_t G_t\nabla_\theta\log\pi_\theta(a_t|s_t)`
(here, using :math:`v_\phi(s,a)` instead of\ :math:`G_t`) "

**CEM in practice**

" sample :math:`\{\tau_i\}_{i=1}^N \sim \pi_\theta(a_t|s_t)` where
:math:`{ \tau = \{ s_t,a_t ... s_T\} }` sort best performing
:math:`\tau` using :math:`v_\phi` Update value approximation
:math:`v_\phi` using :math:`\phi \leftarrow \phi + \nabla_\phi J(\phi)`
(need to define :math:`J(\phi)` ) for each :math:`\tau` ----for each
timestep :math:`t=\{0,1..T-1\}`:
--------:math:`G_t = \sum_{k=t}^T\gamma^k r(s_k,a_k)`
:math:`\min \nabla_\theta J(\theta) = -\sum_\tau\sum_t G_t\nabla_\theta\log\pi_\theta(a_t|s_t)`
(here, using :math:`v_\phi(s,a)` instead of\ :math:`G_t`) "

.. figure:: https://imgur.com/ggLH7UD.png
   :alt: 

Cross entropy method
~~~~~~~~~~~~~~~~~~~~

**Psuedo-code**: (`source <https://arxiv.org/pdf/1909.12830.pdf>`__)

" sample :math:`{ \{ x_{i,t} \} }_{i=1}^N \sim g_{\phi_t}(.)` evaluate
:math:`v_{t,i} := f_\theta(x_{t,i})` sort top samples refit
:math:`g_\phi` with top samples:
:math:`\phi_{t+1}=argmax_\phi \sum_i \mathbf{1}\{ P \} \log g_\phi(x_{i,t})`

"

where :math:`P` is a condition, if true consider the sample :math:`i`,
here the condition is putting a threshold on the reward. Sample
:math:`(s,a)` pairs if :math:`r(s,a)\geq threshold`.

**Code**: \* `CEM implementation on
CartPole-v0 <https://github.com/Neo-47/CE-methods-on-gym-environments/blob/master/cartpole.py#L100>`__
\* NOTE:
(`source <https://gist.github.com/domluna/022e73fd5128b05bdd96d118b5131631>`__)
Preliminary investigation showed that applicability of CE to RL problems
is restricted severly by the phenomenon that the distribution
concentrates to a single point too fast. To prevent this issue, noise is
added to the previous stddev/variance update calculation.

**NOTE**: \* It's a non-parametric approach (not gradient based like
most policy search) \* Here's new paper which proposes "Differentiable
CEM" https://arxiv.org/pdf/1909.12830.pdf

**Reads**: \* `Reference shared by
pierre-luc <https://photos.google.com/share/AF1QipODcxqYaw37s5d39PCZGX85HY4BsX9vrn8KnT_Q2KJ8L7awCKCPhCZybX8iYtc8XA?key=NGVSNHFzSGFGU0pabjBzVWZJdThZUEN5SVc5UEp3>`__
\* `The Cross Entropy method for Fast Policy
Search <https://www.aaai.org/Papers/ICML/2003/ICML03-068.pdf>`__ (**need
to check**) \* Proposes an smoothing update:
:math:`\phi_{t+1}=\alpha\phi_{t+1}+(1-\alpha)\phi_t` \* `A Tutorial on
the Cross-Entropy
Method <https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf>`__

**Good Repo** \*
https://github.com/eleurent/rl-agents#cem-cross-entropy-method (**need
to check**)

**TODO** \* Compare CEM with REINFORCE \* CEM + Smoothing update
(`source <https://www.aaai.org/Papers/ICML/2003/ICML03-068.pdf>`__) +
adding noise
(`source <https://gist.github.com/domluna/022e73fd5128b05bdd96d118b5131631>`__)
+ regularizer

Summary of meetings
-------------------

August 26
~~~~~~~~~

-  Moksh, restart buffer: problems with the restart buffer, using Morgan
   Fingerprints to add the buffer or reject, with buffer pruning after a
   certain size. Problem: no exploration -> start the episode 50% of the
   time from the buffer, 50% from a random walk, this improves things
   quite a bit. Could explore parameter :math:`p`, currently 50% in
   conjuction with similarity threshold.

   -  Yoshua: molecules are pruned based on reward, but inserted based
      on similarity, perhaps we should also use the reward/energy to
      accept/reject molecules. Could also keep track of a mean/variance
      estimate of Value. Compare stored gaussians with incoming
      molecules.
   -  Sanity check experiment on threshold: high thresholds is
      equivalent to adding blindly with p=50%
   -  Why does always adding to the buffer hurts? This probably hurts
      exploration, we'd end up seeing molecules twice as much as
      necessary.
   -  On initial state distribution: Pierre-Luc this could be a problem,
      Yoshua: well the buffer could be a statistic of the "search
      state", Doina: this might not even be an RL problem so it may not
      matter. You can start a search from anywhere.
   -  Yoshua: we should aim to understand this system as a search and
      the buffer as the state. Pierre-Luc: the "RL" algorithms come with
      certain assumptions, if we're not careful about the assumptions
      we're making. If they don't then it shouldn't be too surprising.
   -  Yoshua could sample from softmax instead of uniformly. Could also
      use expected improvement, Thompson sampling.
   -  Should the value be a max? Then the uncertainty model will be
      different. Emmanuel: but the tail of the max is very noisy (from
      the fat tail) so this may not be very useful. In experiments not
      very useful after some number of molecules (check with Bogdan?)

-  Moksh, uncertainty prediction:

   -  Why is the predicted uncertainty so smooth? Not enought
      capacity/data? Doina: test this hypothesis with more points.
   -  There's a problem with measure of true epistemic uncertainty. Need
      to take difference between
      :math:`(f_\theta(x) - \mathbb{E}_\eta[f(x;\eta)])^2` (:math:`\eta`
      the noise)
   -  MC Dropout almost done
   -  When this works for a stationary setting, then we can move to
      non-stationary, active learning.

-  Victor: 7m molecule dataset, sample proportional to the energy, does
   it get better. Initially this did not outperform uniform sampling:

   -  Limiting factor in RL: MPNN is not good on high-reward molecules.
   -  This may be fixed by changing the training objective, e.g.
      selecting for regret, true ranking of molecules.
   -  Oversampling methods do not seem to improve regret.
   -  Now, checking effect of changing from uniform to gaussian
      sampling, how does top-15 top-50 regret changes?
   -  Doina: we don't only care about the ranking, at the high end the
      energies become extreme and we care about the values. An error of
      .2 is not the same in the middle than in the high end. Literature
      on search engines might be interesting.
   -  We could train a bunch of MPNN on each bin of energy. Then you
      need an MPNN to say which bin to send the molecule to.

-  How should we train the reward neural net? Ideally, with respect to
   the distribution it will see when it is used. E.g. the kind of visits
   a search/RL algorithm would see. Uniform might be a proxy but not the
   best.

August 19 8:30
~~~~~~~~~~~~~~

-  Maksym will focus on fixing a bug which should improve the reward
   learning (not much details)
-  Moksh: he verfied if he can recover the PPO performance when he sets
   alpha to 0, also working on a bug
-  Moksh: improve plots and will add legends, some issues with reward
-  Moksh: instability in learning, maybe due to LR
-  Moksh: try shorter runs, models are flattened, increase number of
   actors
-  Moksh: evaluate using docking once in a while
-  Use same metrics as Emmanuel
-  Moksh: in uncertainty learning, make sure you have samples in outside
   regions, use estimate of density as feature, i.e. MC dropout, picking
   a random molecule can help sampling OOD, another idea is to use GT as
   benchmark...
-  Emmanuel: distributed alphazero
-  Organization of project: Adding a spreadsheet for ideas, extend the
   hackmd for whole project

.. |image0| image:: https://i.imgur.com/IINBYAz.png
.. |image1| image:: https://i.imgur.com/2RmgeBg.png
.. |image2| image:: https://i.imgur.com/ZyMDLKR.png
.. |image3| image:: https://i.imgur.com/7WeXxsz.png
.. |image4| image:: https://i.imgur.com/aRuoeVD.png
.. |image5| image:: https://i.imgur.com/hM3ivvP.png
.. |image6| image:: https://i.imgur.com/BCcJeYc.png
.. |image7| image:: https://i.imgur.com/Yls1Ucf.png
.. |image8| image:: https://i.imgur.com/GKQZDE5.png
.. |image9| image:: https://i.imgur.com/YDwhOhF.png
.. |image10| image:: https://i.imgur.com/YKaocj1.png
.. |image11| image:: https://i.imgur.com/3ED5dBK.png
.. |image12| image:: https://i.imgur.com/NAqE1wN.png
.. |image13| image:: https://i.imgur.com/YQJYDFG.png
.. |image14| image:: https://i.imgur.com/oO1J1cm.png
.. |image15| image:: https://i.imgur.com/40w1rS6.png
.. |image16| image:: https://i.imgur.com/MrCAIOU.png
.. |image17| image:: https://i.imgur.com/wiJ8l8z.png
.. |image18| image:: https://i.imgur.com/CM2OQh2.png
.. |image19| image:: https://i.imgur.com/oSTIahh.png
.. |image20| image:: https://i.imgur.com/MYCpWNq.png
.. |image21| image:: https://i.imgur.com/AgZ4X0I.png
.. |image22| image:: https://i.imgur.com/MS6ZoGs.png
.. |image23| image:: https://i.imgur.com/cxO9pB5.png
.. |image24| image:: https://i.imgur.com/fFTfV53.png
.. |image25| image:: https://i.imgur.com/2crB8ay.png
.. |image26| image:: https://i.imgur.com/xKMhZqB.png
.. |image27| image:: https://i.imgur.com/x2FPaVI.png
.. |image28| image:: https://i.imgur.com/RMN8Wyk.png
.. |image29| image:: https://i.imgur.com/GLDODhb.png
.. |image30| image:: https://i.imgur.com/0qRdwP8.png
.. |image31| image:: https://i.imgur.com/tHqmLzm.png
.. |image32| image:: https://i.imgur.com/Kve4Nvr.png
.. |image33| image:: https://i.imgur.com/LBQT05k.png
.. |image34| image:: https://i.imgur.com/4bfxcP0.png
.. |image35| image:: https://i.imgur.com/hRE7cgG.png
.. |image36| image:: https://i.imgur.com/zASFhP6.png
.. |image37| image:: https://i.imgur.com/TDsHhj2.png
.. |image38| image:: https://i.imgur.com/MpZ0ox4.png
.. |image39| image:: https://i.imgur.com/JwgTaU5.png
.. |image40| image:: https://i.imgur.com/TlnMSLb.png
.. |Top 15 and Top 50 regret of three different dropout cases on 260k dataset| image:: https://i.imgur.com/y15MItk.png
.. |Results from running grid search of kappa on 260k dataset twice| image:: https://i.imgur.com/iGK1oTX.png
.. |Results from random baseline and grid search of kappa on 260k dataset| image:: https://i.imgur.com/4VChU0Q.png
