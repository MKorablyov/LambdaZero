************************************
Multi-Fidelity Search with MF-CP-UCB
************************************

This work is based on the following paper: `Link <https://papers.nips.cc/paper/6118-gaussian-process-bandit-optimisation-with-multi-fidelity-evaluations.pdf>`

We want to address the problem of molecule search over a large space of molecules for one which maximizes a given scoring function.
These functions are available in multiple forms of varying costs and accuracy.
For example, the most accurate score would come from wet lab experiments, but those are very expensive in terms of monetary costs and time.
At the other extreme, we can learn estimates of these scores, which are quick to run, but much less accurate.
Ideally, we would be able to make use of all sources of information to come up with the best molecule possible within a given time and resource constraint.

#########
MF-GP-UCB
#########

The MF-GP-UCB algorithm is an upper confidence bound algorithm which learns an approximation of each each oracle with a Gaussian process, and optimistically chooses states to evaluate in order to maximize the UCB.
The Gaussian process for fidelity :math:`i\in\{1,2,\cdots,M\}` at time :math:`t` can be expressed with a mean function :math:`\mu^{(i)}_t(s)` and its standard deviation :math:`\sigma^{(i)}_t(s)`.
The UCB for a given fidelity level is 
..:math::
        \varphi^{(i)}_t(s) = \mu_t^{(i)}(s')+\beta^{1/2}\sigma_t^{(i)}(s')+\zeta^{(i)}
where :math:`\beta` is a hyperparameter which determines the level of confidence we want to attain, and :math:`\zeta` is a hyperparameter which bounds the absolute difference between fidelity levels (i.e. :math:`||f^{(i)}-f^{(i-1)}||_\infty \leq \zeta` where :math:`f^{(i)}` is the output of an oracle of fidelity :math:`i`).

It is very unlikely (how unlikely is determined by $\beta$) for the true value of $s$ to be higher than any of the UCB estimates, so the most optimistic choice is given by
..:math::
        s_t = \arg\max_s \left\{ \min_i \varphi_t^{(i)}(s) \right\}

Once the state is chosen, the agent needs to decide on which oracle to query.
The algorithm starts off at the lowest fidelity oracle, and queries the first oracle to satisfy
..:math::
        \beta^{1/2}\sigma_t^{(i)}(s_t) \geq \gamma^{(i)}
The :math:`\gamma^{(i)}` parameter is doubled if ever oracle :math:`i` has not been queried for over :math:`\lambda^{(i+1)}/\lambda^{(i)}` steps.

They prove that this algorithm outperforms (as measured by cumulative regret) the single-fidelity version of the same algorithm, and back up their claim with experimental results.

############################
Extension to Molecule Search
############################

In the scenario we care about, we can't directly sample the space of molecules.
They have to be built through a graph traversal to ensure that the resulting molecule is valid.
The extension to MF-GP-UCB we propose is that instead of directly choosing a state according to equation (\ref{eqn:acquisition-func}), we traverse the graph by going to the state that can eventually lead to a state that optimizes for the same criterion.

Secondly, the performance measure we care about is not cumulative regret, but rather instantaneous regret whenever the search is stopped.
So maximizing the UCB score is not necessarily optimal for us.
For evaluation purposes, we experiment with choosing states to optimize for one of two criteria: the mean estimate of the highest fidelity oracle :math:`\mu^{(M)}`, or the UCB score.
The chosen molecule is one which maximizes the chosen criterion within a trajectory formed by greedily choosing actions with respect to this same crierion.
