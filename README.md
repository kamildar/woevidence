# Weight of evidence transformation

## How to
clone this repository into your working directory
```
git clone https://github.com/kamildar/woe_transform.git
```
after that use classes ```WoeTree``` or ```WoeNN```.

Example:
```python
woe = WoeTree(criterion='entropy', max_depth=4, n_jobs=-1)
woe_data = WoeTree.fit_transform(features, target)
````
and use ```woe_data``` as features for task.

## More information
Additional information on encoders will be provided as soon as possible.