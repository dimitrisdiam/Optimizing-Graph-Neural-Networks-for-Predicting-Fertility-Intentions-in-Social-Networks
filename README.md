# Predicting Fertility Intentions with Graph Neural Networks

Master's thesis, Applied Data Science, Utrecht University. By Dimitrios Diamantidis.

## What this is

Each person in the study has an egocentric social network: themselves plus the people around them, with attributes for every person and the relationships between them. The question is whether you can predict a person's intention to have children from the structure and attributes of that network.

To answer it, I build one graph per participant and train a Graph Convolutional Network to classify their fertility intention. The project tests every combination of node features to see which ones actually help the prediction.

## Approach

- One graph per participant, with the participant added as a central node connected to everyone in their network.
- A three layer GCN with mean pooling, built in PyTorch Geometric.
- K-fold cross validation (5 folds) with a held out test set.
- Oversampling applied to the training folds only, to handle class imbalance without leaking into validation.
- Early stopping on validation F1, keeping the best weights.
- Evaluation with macro F1 and a confusion matrix, since the classes are imbalanced and accuracy alone would be misleading.

The model is run across all feature combinations, and each is scored so the most useful set of features can be identified.

## Results

Macro F1 on the held out test set, for a sample of feature combinations.

| Features used | Test macro F1 |
|---------------|---------------|
| age, num_children, sex, child_free, relationship_type, non_f2f | 0.526 |
| num_children, child_free, friend, f_to_f, relationship_type, help | 0.510 |
| age, num_children, sex, relationship_type, help, non_f2f | 0.463 |
| age, num_children, sex, child_free, friend, relationship_type, non_f2f | 0.423 |
| num_children, sex, f_to_f, relationship_type, help | 0.402 |

Macro F1 sat in a fairly narrow range, roughly 0.40 to 0.53, across the
combinations tested. `num_children` and `relationship_type` appeared in every
stronger combination, which suggests they carried much of the usable signal. Age
helped in the best run but was not essential, since a strong combination without
it scored almost as high. The overall picture is that fertility intention was only
partly predictable from network attributes here, with no single feature dominating
and the model staying well above a random baseline but short of high accuracy.

## Background

The work builds on Stulp et al. (2023), which studied how individual traits and social network structure relate to fertility decisions. This thesis applies Graph Neural Networks to the same kind of personal and egocentric network data to predict fertility intentions.

## Data

Personal and egocentric network data, with one network per participant. Each
participant folder holds the network edges and the attributes of the people in
it. Survey answers are messy by nature, so a large part of the work is the
preprocessing that turns answers like "50+", "Expecting first child" and
"I don't know" into clean numeric features.

## Files

```
├── thesis_gnn.py    Full pipeline: preprocessing, model, training, evaluation
├── requirements.txt Dependencies
└── README.md
```

## How to run it

```bash
pip install -r requirements.txt
```

Place the participant folders under `participants/` and the participant table as
`data_subset.csv` in the same directory, then run:

```bash
python thesis_gnn.py
```

The script trains across all feature combinations and prints the test F1 for each,
followed by the best combination.

## Note on the data

The dataset is not included in this repository because it is based on restricted
survey data. The code shows the full method end to end.
