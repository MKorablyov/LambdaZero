### Move docking to the GPU
Install and make work AtoDock-GPU

### Reduce aleatoric uncertainty
Repeat docking several times take max - we discovered there is a lot of aleatoric uncertainty

### Finetune
With some experimental data from ChEMBL with scaffold split

### DEL data
Obtain DEL data from ZebiAI or Anagenex or someone else

### DEP data
get FEPs from DESRES

### Multiple protein conformations
At smalest these could be taken from PDB and we could test on several structures. Maybe, a better docking could be 
trained with more realistic physics and multiple protein conformations. 

### Evaluation
Evaluate any oracle/ML addition/model on the test set from ChEMBL
ADMET and more serious drug properties