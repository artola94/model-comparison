# Credit Risk Scoring: TabFM vs. Gradient Boosting

This project compares Google's **TabFM** (a tabular foundation model that predicts by in-context learning) with **Catboost** on serious-delinquency scoring, using **Give Me Some Credit**.

TabFM reads labelled rows as context in a single forward pass. The study therefore has two arms, reported separately:

1. **Matched context.** Both models see the same stratified sample of (N \\in {100, 500, 1000}) rows, repeated 30 times.
2. **Full pool.** Catboost trains on the full training pool (\~120,000 rows). TabFM ensembles 32 sampled contexts from that same pool.

Both arms are scored on one frozen, stratified test set of 30,000 rows. The run lives in Google Colab on the PyTorch backend.