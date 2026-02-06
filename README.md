# PNAS: Protein-Nucleic Acid Binding Site Prediction Model

## Important Notes
1. Source code for the paper *Protein-nucleic acid binding site prediction based on knowledge-guided prompt learning and semi-supervised contrastive learning on sequences* (JCIM submission).
2. Raw dataset: Publicly available from GraphBind official release (http://www.csbio.sjtu.edu.cn/bioinf/GraphBind/); data processing see Section 4.1 of the manuscript.
3. Derived features can be fully reproduced by running the feature extraction scripts with manuscript-specified parameters on the GraphBind dataset.

## Directory Structure
- **PDNA1/**, **PRNA/**,**esm/**: Contains all required feature files generated from the dataset.
- **script/**: Includes feature generation scripts, model scripts, scripts for calculating model evaluation metrics, loss function scripts, and running logs.
- **model/**: Stores various model parameters. Among them, ml0-ml4 are the results of 5-fold cross-validation; the file *ml_pu* contains the model parameters after training with PU learning integrated. The difference between *\*_loss* and *\*_mcc* files is that the saved training results use loss value or MCC value as the criterion for evaluating model performance, respectively.
