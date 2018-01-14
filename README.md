# woe_transform
Weight of evidence transformation

Functions apply tree or random forest classifier to dataset for getting terminal leafs and after that apply WoE transformation.
Current problems:
- no specific way for handling missing values
- returns inf for leafs with only 1 or 0 values, tune tree or random forest parameters to avoid this
