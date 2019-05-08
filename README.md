# woevidence
Module for weight of evidence transformation

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
woe = WoeTree(criterion='entropy', max_depth=4, n_jobs=-1)
woe_data = woe.fit_transform(features, target)
````

### Advantages
- missing values handling strategies, argument `na_strategy` allows put NA's in its own bucket or interpret them as the worst/best group (leaf)
- `gini` or `entropy` or custom splitting criterion
- flexible parameters for buckets (leaves): `min_samles_leaf`, `min_samples_class`, `max_depth `
