# woevidence
Module for weight of evidence transformation. More on WOE encoding [here](https://medium.com/@sundarstyles89/weight-of-evidence-and-information-value-using-python-6f05072e83eb).

## Getting started
### Installing
```bash
git clone https://github.com/kamildar/woe_transform.git
cd woe_transform
pip install -e .
```

### Usage
```python
from woevidence import WoeTree
woe = WoeTree(criterion='entropy', max_depth=4, n_jobs=-1, 
              categorical_features=["education", "city"])
woe_data = woe.fit_transform(features, target)
````

### Advantages
- missing values handling strategies, argument `na_strategy` allows to put NA's in special bucket or interpret them as the worst/best group
- categorical features support
- `gini` or `entropy` or custom splitting criterion
- flexible parameters for buckets (leaves): `min_samles_leaf`, `min_samples_class`, `max_depth`
- scikit-learn compatibility