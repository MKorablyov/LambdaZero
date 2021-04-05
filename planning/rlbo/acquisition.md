### Evaluation of uncertainty
* **Plots**
    * Rank Correlation
    * Error Correlaiton
    * LogLik
    * calibration plot as Leo made

* **Test set**  
We could take one from either Boltzmann Search/Maximum Likelihood Model
    * Systematic Generalization (very good molecules)
    * Scaffold Split

* **Runtime**
    * Plug into Acquisition Function, Boltzmann Search
    * Plug into Acquisition Function, PPO


### Fast and Slow updates
MPNN + linear regression or GP on MPNN embedding (could be simple GP, could be DEUP etc.)


### Efficient with acquisition 
Keep some previously proposed somewhat good molecules / re-evaluate these every time


### Uncertainty Model Types
Ensemble  
MC Dropout  
Structured MC Dropout (aleviated problem with samples in MC Dropout correlation)  
Direct Error Prediction (baseline)  
Evidential Regression MIT/ICML  
DEUP: MPNN + (Dropout / Ensemble) + spectral norm + GP  


### Diverse distribution
Autoencoder on Zinc add KL term as loss
Volume of convex set of generated molecules or even distance to nearest neighbor as approximation
Sum{i,j}sigmoid(tanimoto)

### Acquisition functions
EI
MES
UCB