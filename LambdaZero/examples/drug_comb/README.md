# Predicting the effect of drug combinations: an Active Learning Approach

## Databases

- We use [drugcombDB](https://academic.oup.com/nar/article/48/D1/D871/5609522). Percentage of cell death as well as 4 synergy scores (HSA, Loewe, ZIP, Bliss) are available. **6M** unique experiments, **60k** scores.

- For now we usePPI and drug targets available in **drugcombDB** (retrieved from **Drugbank**, **Stich** and others).

- We want to use cleaner databases ([SNAP](http://snap.stanford.edu/), [BindingDB](https://www.bindingdb.org/bind/index.jsp), [Human Interactome](https://www.nature.com/articles/s41586-020-2188-x)) which are less biased towards highly studied drugs/proteins

## Structure of the Dataset

A graph $G=(V, E)$ where the vertices are either drugs or proteins: $V = V_d \cup V_p$.

- If a vertex $v \in V_d$ is a drug, it is associated with a feature representation $x^v \in \mathbb{R}^k$ containing the *Morgan fingerprint* of the drug. We generated fingerprints with $nBits = 1024$ and $radius=4$

- If a vertex $v \in V_p$ is a protein, there is no feature available. Currently implementing features based on the eigenvectors of Laplacian operator.

- The edges $E$ represent drug-drug interactions, drug-protein interactions and protein-protein interactions. We can consider these edges to be of three types $E = E_{dd} \cup E_{dp} \cup E_{pp}$.

- Each edge is associated with a feature vector $x^e \in \mathbb{R}^4$. If an edge corresponds to a drug-drug interaction, $e \in E_{dd}$, $x^e$ contains the 4 drug synergy scores, otherwise $x^e = 0_4$.

- **Note**: protein features and drug features are not expected to have the same dimensionality.

## Giant Graph pipeline

### Model

**Overview**: Graph Convolutional Network + score predictor. The GCN uses multiple types of messages between the different types of nodes.

Initialize matrices $M_{p \leftrightarrow p}$, $M_{d \rightarrow p}$, $M_{p \rightarrow d}$, $M_{d}$, $M_{p}$. In what follows, $\sigma$ denotes any activation function such as ReLU or Sigmoid.

 **for layer l from 1 to L, iterate**:

if $v \in V_d$:
| $h^v_{l+1} = \sigma(M_d h^v_{l} + \sum\limits_{\omega \in N(v) \cap V_p}M_{p \rightarrow d} h^\omega_l)$ |
|----------|

Note: we could use concatenation instead of sum between node and neighbour's messages

 if $v \in V_p$:
 | $h^v_{l+1} = \sigma(M_p h^v_{l} + \sum\limits_{\omega \in N(v) \cap V_d}M_{d \rightarrow p} h^\omega_l + \sum\limits_{\omega \in N(v) \cap V_p}M_{p \leftrightarrow p}h^\omega_l)$ |
 |----------|

 **After T time steps**:

 Predict the drug synergy score with a function $f_\phi(h^{v1}_L, h^{v2}_L, x^{v1}, x^{v2})$. We can choose something simple at first: $f_\phi = h^{v1}_L G h^{v2}_L$ . Note: in the case where $G$ has negative eigenvalues, drugs with opposite embeddings will have a high predicted synergy score.

### Short term

- Implementing features based on the eigenvectors of Laplacian operator.
- Implement simple baseline
- Hyperparam tunning
- Integrate with Active Learning

### Future directions

- Use Graph Attention Networks
- Investigate how to share knowledge between cell types in a smart way
- Use percentage of cell deaths as input


## Baseline pipeline

Let $\phi: (fingerprint, cell-line)\rightarrow \mathbb{R}^k$ be a MLP. (we choose $k=100$).

For a given pair of drugs with fingerprints $fp_1$ and $fp_2$, whose score was acquired on cell line $c$, we predict the score as:

| $\hat{score} = \langle \phi(fp_1, c) | \phi(fp_2, c) \rangle$ |
|----------|

## SubGraphs pipeline

TODO

## Active Learning

Let $Q$ denote the set of drug pairs that we have queried so far. We can choose this set to be non empty at first, and *pretrain* the model with these pairs.

**Bayesian Learning**:

We want to get $N$ samples of predicted scores for each drug pair by introducing noise in the model and performing several forward passes (in the spirit of Monte Carlo dropout). Two proposed ways to do this:
- 1 forward pass in the GNN + $N$ forward passes in $f_\phi$ with different dropout configurations
- $N$ forward passes in the GNN with different random initializations of the embeddings $h$.

**Acquisition function**

We use *Expected Improvement*

Let $s_{+}$ be the best score *seen* so far among all drug pairs. For each drug pair $dd$, we get $N$ samples of predicted synergy scores $s_{dd}^1, ...,s_{dd}^{N}$.

The Expected Improvement for the drug pair $dd$ is:

| $EI = \mathbb{E}\max(s_{dd} - s_{+}, 0)$ |
|----------|

It can be approximated by:

| $\hat{EI} = \frac{1}{N}\sum \sigma(s_{dd}^i - s_{+})$ |
|----------|

where $\sigma$ denotes the ReLU function.

Then we query the pairs with highest (one at a time or several at a time) Expected Improvement $\hat{EI}$. The query at iteration $j$ is $Q_j$.

We expend $Q$ with the newly queried pairs: $Q \leftarrow Q \cup Q_j$ and train the model on all previously queried pairs $Q$.

## References

- MPNN: [link](https://arxiv.org/pdf/1704.01212.pdf)

- Review on GNNs: [link](https://arxiv.org/pdf/1812.08434.pdf)

- Decagon: [link](https://academic.oup.com/bioinformatics/article/34/13/i457/5045770)

- Bayesian Optimization: [link](http://krasserm.github.io/2018/03/21/bayesian-optimization/#:~:text=Expected%20improvement%20is%20defined%20as,if%20%CF%83(x)%3D0)

- Graph Laplacian: [link](https://csustan.csustan.edu/~tom/Clustering/GraphLaplacian-tutorial.pdf)
