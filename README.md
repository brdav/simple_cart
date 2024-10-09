<div align="center"> 

# Decision tree growing and pruning in NumPy

![Education](https://img.shields.io/badge/Education-B9F3Bf.svg)

</div>

Have you ever wondered how exactly decision tree growing and pruning works? Most references (e.g. [1]) skip the implementation details. For CART decision trees (such as the one implemented here), Breiman [2] provides a detailed explanation. It turns out that CART decision trees can be grown and pruned (minimal cost-complexity pruning) with a standard **depth-first** algorithm [3]. See `decision_tree.py` for a dead-simple recursive implementation and the notebook `demo.ipynb` for a demo.

## Requirements

The source code uses Python >= 3.9 with `numpy` as the sole dependency. The package can be installed with:
```
pip install -e .
```

To run the notebooks, the following packages are additionally required:
```
pip install notebook scikit-learn matplotlib
```

Testing additionally requires `pytest`, and can be executed with:
```
python -m pytest
```

## Disclaimer

Note that the goal of this repo is educational, so the focus is on readability and simplicity, rather than efficiency and performance! Also note that a **best-first** algorithm for growing decision trees is sometimes preferred as it enables early stopping.

## References

[1] Hastie, T., et al. "The Elements of Statistical Learning: Data Mining, Inference, and Prediction." New York: Springer, 2009

[2] Breiman, L. "Classification and regression trees." Routledge, 2017

[3] Cormen, T. H., et al. "Introduction to Algorithms." MIT press, 2022