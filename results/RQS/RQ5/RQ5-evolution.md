
### What's the purpose of the notebook?

We want to ensure the robustness of the estimation of the IS Score, to be sure that even with few measurements, we can be confident that the IS score actually gives a good indication about the level of input sensitivity of a couple of software-performance property. So in this notebook, we vary the amount of data available and compute the IS score to compare its values according to the available dataset.

### Results 

- Evolution is quite stable, which is a good result. With few data, we are quite close to the final value estimated with many measurements. We explain it by construction of the indicator; since we choose the minimal and maximal values for the first part of the formula, it converges quickly to the final value.
