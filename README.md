# woevidence
Module for weight of evidence transformation.

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