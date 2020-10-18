# Summary
This directory holds the code for predicting synergistic drug combinations.  The code revolves around the DrugComb 
database (https://drugcomb.fimm.fi/, https://academic.oup.com/nar/article/47/W1/W43/5486743), as well as various
predictive models, most centered around MPNNs.  The code is organized into three separate main directories, while
the entry point for the code coming in `train.py`.  In the following, we will first summarize how to run the code,
and subsequently the structure of the codebase.

## Running the Code
The entry point for the code is the file `train.py`.  A central tenet of this code is that, while a number of 
different models are contained within it, they all conform to the same interface and so are simple to plug and
play in the `train.py` file.  The avenue through which one should interact with the training procedure is via
the configuration dictionaries in `train.py`, namely the `model_config`, `predictor_config`, `pipeline_config`, 
and `dataset_config`.  

### Models
The `model_config` contains parameters for the the model used for learning representations
of the graph.  This includes the model type itself.  These models can be found in the file `models/models.py`.
The `torch.nn.Module` subclasses in the `models/models.py` file are the MPNNs that have been written 
thus far.  Available models are the `Baseline` model, and the `GiantGraphGCN` model.  The `Baseline` model
does not do any interesting representation learning and immediately plugs the graph's base features into a
predictor.  The `GiantGraphGCN` is an MPNN on top of the giant graph (we define here the giant graph as the
union of the drug drug interaction, drug protein interaction, and protein protein interaction graphs).  It
has many different options, including whether or not to use LRGA (https://arxiv.org/pdf/2006.07846.pdf), 
which edge types to pass messages along, and whether or not to backprop through the MPNN only periodically.

### Predictors
The model specified by `model_config` only serves to build learnt representations of the vertices of the
drug and protein graphs.  After we find these representations, we still need to craft a prediction from
these representations.  The dictionary `predictor_config` contains parameters for the predictor that acts
on top of the MPNN's learned reprersentations.  So far, we have generally used simple MLPs for final prediction.
These classes are found in the file `models/predictors.py`, and the `predictor_config` should specify them
as well as the various parameters required by the predictors in `models/predictors.py`.


