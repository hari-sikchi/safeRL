# Safe Reinforcement Learning Algorithms   

## HCOPE

High-Confidence Off-Policy Evaluation.

Python Implementation of HCOPE lower bound evaluation as given in the paper:
Thomas, Philip S., Georgios Theocharous, and Mohammad Ghavamzadeh. "High-Confidence Off-Policy Evaluation." AAAI. 2015.

### Requirements
* PyTorch
* Numpy
* Matplotlib
* scipy
* gym

### Running Instructions

1. Modify the environment in the main function, choosing from  OpenAI gym. (Currently the code works for discrete action spaces)    
2. Run python hcope.py   

### Notes

* To reproduce the graph given in the original paper explaining the long tail problem of importance sampling, use the 
```
visualize_IS_distribution()
```
method.   
* To estimate c*, I use the BFGS method which does not require computing hessian or first order derivative.   
* The ```hcope_estimator()``` method also implements a sanity check, by computing the discriminant of the quadratic in parameter delta(confidence). If it does not satisfy the basic constraints, the program prints the bound predicted is of zero confidence.   
* The random variables are implemented using simple importance sampling. Per-decision importance sampling might lead to better bounds and is to be explored.   
* A bilayer MLP policy is used for general problems.   




## Importance Sampling

Implementation of:    
* Simple Importance Sampling   
* Per-Decision Importance Sampling    
* Normalized Per-Decision Importance Sampling (NPDIS) Estimator    
* Weighted Importance Sampling (WIS) Estimator   
* Weighted Per-Decision Importance Sampling (WPDIS) Estimator    
* Consistent Weighted Per-Decision Importance Sampling (CWPDIS) Estimator   
    
Comparision of different importance sampling estimators:   
![Different Importance sampling estimators](https://github.com/hari-sikchi/HCOPE/blob/master/importance_sampling/importance_sampling.png)   

 Image is taken from phD thesis of P.Thomas:    
 Links: https://people.cs.umass.edu/~pthomas/papers/Thomas2015c.pdf   

