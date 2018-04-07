# Weight of evidence transformation

## How to
Example:
```python
woe = WoeTree(criterion='entropy', max_depth=4, n_jobs=-1)
woe_data = WoeTree.fit_transform(features, target)
````
and use ```woe_data``` as features for task.

## More information
Additional information on encoders will be provided as soon as possible.