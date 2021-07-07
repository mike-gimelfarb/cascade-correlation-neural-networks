# cascade-correlation-neural-networks
A general framework for **building and training constructive feed-forward neural networks**. Provides an implementation of sibling-descendant CCNN (Cascade-Correlation) [1,2] with extendable wrappers to tensorflow, keras, scipy, and scikit-learn. Also supports custom topologies, training algorithms, and loss functions [3, 4].

# Installation
The simplest way to install this package currently is to clone the repository and use pip. First, clone the repository:

```console
git clone https://github.com/mike-gimelfarb/cascade-correlation-neural-networks.git
```

Next, navigate to the folder and use pip

```console
cd cascade-correlation-neural-networks
pip install .
```

We are currently in the process of hosting this project from PyPI, please stay tuned.

# Requirements
The package has been tested using:
- Python 3.7
- Tensorflow 2.3.1
- scikit-learn 0.23.2
- pandas 1.1.3
- scipy 1.5.2

# Features
Regression
<p align="center">
  <img src="https://github.com/mike-gimelfarb/cascade_correlation_neural_networks/blob/main/examples/images/regression.jpg?raw=true"/>
  <img src="https://github.com/mike-gimelfarb/cascade_correlation_neural_networks/blob/main/examples/images/quantile_regression.jpg?raw=true"/>
  <img src="https://github.com/mike-gimelfarb/cascade_correlation_neural_networks/blob/main/examples/images/bayesian_regression.jpg?raw=true"/>
</p>

Classification
<p align="center">
  <img src="https://github.com/mike-gimelfarb/cascade_correlation_neural_networks/blob/main/examples/images/spirals.jpg?raw=true"/>
  <img src="https://github.com/mike-gimelfarb/cascade_correlation_neural_networks/blob/main/examples/images/spirals_classification.jpg?raw=true"/>
</p>

Unsupervised Learning
<p align="center">
  <img src="https://github.com/mike-gimelfarb/cascade_correlation_neural_networks/blob/main/examples/images/reconstruction.jpg?raw=true"/>
</p>

# References
<ol>
  <li>Fahlman, Scott E., and Christian Lebiere. "The Cascade-Correlation Learning Architecture." NIPS. 1989.</li>
  <li>Baluja, Shumeet, and Scott E. Fahlman. Reducing network depth in the cascade-correlation learning architecture. CARNEGIE-MELLON UNIV PITTSBURGH PA SCHOOL OF COMPUTER SCIENCE, 1994.</li>
  <li>Kwok, Tin-Yau, and Dit-Yan Yeung. "Bayesian regularization in constructive neural networks." International Conference on Artificial Neural Networks. Springer, Berlin, Heidelberg, 1996.</li>
  <li>Kwok, Tin-Yau, and Dit-Yan Yeung. "Objective functions for training new hidden units in constructive neural networks." IEEE Transactions on neural networks 8.5 (1997): 1131-1148.</li>
</ol>

# See Also
<ol>
  <li>https://www.psych.mcgill.ca/perpg/fac/shultz/personal/Recent_Publications_files/cc_tutorial_files/v3_document.htm</li>
</ol>
