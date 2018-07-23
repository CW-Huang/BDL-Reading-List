This page maintains a collection of papers/resources in different categories related to Bayesian Deep Learning & Deep Bayesian Learning (see [YW Teh's talk](https://www.youtube.com/watch?v=LVBvJsTr3rg "NIPS talk") on the dichotomy). 


## Some books
* PGM [Probabilistic Graphical Models: Principles and Techniques](https://mitpress.mit.edu/books/probabilistic-graphical-models), Koller and Friedman 2009
* PRML [Pattern Recognition and Machine Learning](https://www.springer.com/us/book/9780387310732), Bishop 2006


## Core
A generic formula for models with latent variables:
* PGM Chapter 19

Markov Chain Monte Carlo (MCMC) theory and classic algorithms: 
* PGM Chapter 12
* PRML Chapter 11

Hamiltonian Monte Carlo (HMC):
* [MCMC using Hamiltonian dynamics](https://arxiv.org/abs/1206.1901), Neal 2012
* [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/abs/1701.02434), Betancourt 2017

Expectation Maximization (EM) and Variational Inference (VI):
* PRML Chapter 9, 10.1-10.6
* [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670), Blei et al. 2016
* [Graphical Models, Exponential Families, and Variational Inference](https://www.nowpublishers.com/article/Details/MAL-001), Wainwright and Jordan 2008
* [An Introduction to Variational Methods for Graphical Models](https://link.springer.com/article/10.1023/A:1007665907178), Jordan et al. 1999

Amortized Variational Inference and Reparameterization Trick:
* [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), Kingma and Welling 2013
* [Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/abs/1401.4082), Rezende et al. 2014
* [The Generalized Reparameterization Gradient](https://arxiv.org/abs/1610.02287), Ruiz et al. 2016
* [Inference Suboptimality in Variational Autoencoders](https://arxiv.org/abs/1801.03558), Cremer et al. 2018

Hierarchical Variational Methods:
* [An Auxiliary Variational Method](https://www.semanticscholar.org/paper/An-Auxiliary-Variational-Method-Agakov-Barber/07bbffb1d04d252a471c3a40653849b1c8200ede), Agakov and Barber 2004
* [Hierarchical Variational Models](https://arxiv.org/abs/1511.02386), Ranganath et al. 2015
* [Auxiliary Deep Generative Models](https://arxiv.org/abs/1602.05473), Maaløe et al. 2016
* [Markov chain Monte Carlo and variational inference: Bridging the gap](https://arxiv.org/abs/1410.6460), Salimans et al. 2014
* [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770), Rezende and Mohamed 2015

Variance Reduction in VI:
* [Reducing Reparameterization Gradient Variance]
* [Quasi-Monte Carlo Variational Inference](https://arxiv.org/abs/1807.01604), Buchholz et al. 2018
* [Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference](https://arxiv.org/abs/1703.09194), Roeder et al. 2017



Expectation Propagation (EP):
* PRML Chapter 10.7
* PGM Chapter 11.4
* [Proofs of Alpha Divergence Properties](https://www.ece.rice.edu/~vc3/elec633/proof_alpha_divergence.pdf) (lecture note), Cevher 2008
* [Divergence Measures and Message Passing](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2005-173.pdf), Minka 2005


## Deep State Space Models
* [A Recurrent Latent Variable Model for Sequential Data](https://arxiv.org/abs/1506.02216), Chung et al. 2015
* [Filtering Variational Objectives](https://arxiv.org/abs/1705.09279), Maddison et al. 2017
* [Variational Sequential Monte Carlo](https://arxiv.org/abs/1705.11140), Naesseth et al. 2017
* [Auto-Encoding Sequential Monte Carlo](https://arxiv.org/abs/1705.10306), Le et al. 2017
* [Variational Bi-LSTMs](https://arxiv.org/abs/1711.05717), Shabanian et al. 2017


## Normalizing Flows
* [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770), Rezende and Mohamed 2015
* [Improving Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/abs/1606.04934), Kingma et al. 2016
* [Neural Autoregressive Flows](https://arxiv.org/abs/1804.00779), Huang et al. 2018

## Transfer Learning and Semisupervised Learning

## Representation Learning
* PixelVAE
* Variational Lossy Autoencoder
* Vampprior
* Variational Fair Autoencoder
* [Hierarchical VampPrior Variational Fair Auto-Encoder](https://drive.google.com/file/d/1G9vfra-BEgLAQhhfguagT9m72lCVTfWP/view?usp=drive_web), Botros and Tomczak 2018



## Disentanglement in Deep Representations
* [Emergence of Invariance and Disentanglement in Deep Representations](https://arxiv.org/abs/1706.01350), Achille and Soatto 2017
* [Early Visual Concept Learning with Unsupervised Deep Learning](https://arxiv.org/abs/1606.05579), Higgins et al. 2016
* [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl), Higgins et al. 2017
* [Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942), Chen et al. 2018
* [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599), Burgess et al. 2018


## Memory Addressing as Inference


## Discrete Latent Variable


## Bayesian Deep Neural Networks (Variational Approaches)
* [Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks](https://arxiv.org/abs/1502.05336), Hernández-Lobato and Adams 2015
* [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142), Gal and Ghahramani 2015
* [Variational Dropout and the Local Reparameterization Trick](https://arxiv.org/abs/1506.02557), Kingma et al. 2015
* [Dropout Inference in Bayesian Neural Networks with Alpha-divergences](https://arxiv.org/abs/1703.02914), Yingzhen and Gal 2017
* [Multiplicative Normalizing Flows for Variational Bayesian Neural Networks](https://arxiv.org/abs/1703.01961), Louizos and Welling 2017
* [Bayesian Hypernetworks](https://arxiv.org/abs/1710.04759) Krueger et al. 2017

## Bayesian Deep Neural Networks (MCMC Approaches)


## Deep neural networks = Gaussian Process


## SGD as Approximate Bayesian Inference
* [Stochastic Gradient Descent as Approximate Bayesian Inference](https://arxiv.org/abs/1704.04289), Mandt et al. 2017
* [Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks](https://arxiv.org/abs/1710.11029), Chaudhari and Soatto 2017

## SGD as PAC Bayes Bound Minimization






