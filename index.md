This page maintains a collection of papers/resources in different categories related to Bayesian Deep Learning & Deep Bayesian Learning (see [YW Teh's talk](https://www.youtube.com/watch?v=LVBvJsTr3rg "NIPS talk") on the dichotomy). 




## Some books
* PGM [Probabilistic Graphical Models: Principles and Techniques](https://mitpress.mit.edu/books/probabilistic-graphical-models), Koller and Friedman 2009
* PRML [Pattern Recognition and Machine Learning](https://www.springer.com/us/book/9780387310732), Bishop 2006
* MCTME [Monte Carlo theory, methods and examples](http://statweb.stanford.edu/~owen/mc/), Owen (book in progress)


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

* MCTME Chapter 8, 10

* [Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference](https://arxiv.org/abs/1703.09194), Roeder et al. 2017

* [Reducing Reparameterization Gradient Variance](https://arxiv.org/abs/1705.07880), Miller et al. 2017
* [Quasi-Monte Carlo Variational Inference](https://arxiv.org/abs/1807.01604), Buchholz et al. 2018

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
* MCTME Chapter 4
* [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770), Rezende and Mohamed 2015
* [Improving Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/abs/1606.04934), Kingma et al. 2016
* [Improving Variational Auto-Encoders using Householder Flow](https://arxiv.org/abs/1611.09630), Tomczak and Welling 2016
* [Improving Variational Auto-Encoders using convex combination linear Inverse Autoregressive Flow](https://arxiv.org/pdf/1706.02326), Tomczak and Welling 2017
* [Sylvester Normalizing Flows for Variational Inference](https://arxiv.org/abs/1803.05649), Berg et al. 2018
* [Neural Autoregressive Flows](https://arxiv.org/abs/1804.00779), Huang et al. 2018
* [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803), Dinh et al. 2016

* [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039), Kingma and Dhariwal 2018

## Importance Weighted Autoencoder

- [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519), Burda et al. 2015
- [Reinterpreting Importance-Weighted Autoencoders](https://arxiv.org/abs/1704.02916), Cremer et al. 2017
- [Tighter Variational Bounds are Not Necessarily Better](https://arxiv.org/abs/1802.04537), Rainforth et al. 2018
- [On Nesting Monte Carlo Estimators](https://arxiv.org/abs/1709.06181), Rainforth et al. 2018

## Inference as Adversarial Training

* [Adversarially Learned Inference](https://arxiv.org/abs/1606.00704), Dumoulin et al. 2016
* [Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks](https://arxiv.org/abs/1701.04722), Mescheder et al. 2017
* [Variational Inference using Implicit Distributions](https://arxiv.org/pdf/1702.08235;Variational), Huszar 2017

## Transfer Learning and Semisupervised Learning

* [Semi-Supervised Learning with Deep Generative Models](https://arxiv.org/abs/1406.5298), Kingma et al. 2014
* [Towards a Neural Statistician](https://arxiv.org/abs/1606.02185), Edwards and Storkey 2016
* [One-Shot Generalization in Deep Generative Models](https://arxiv.org/abs/1603.05106), Rezende et al. 2016
* [Uncertainty in Multitask Transfer Learning](https://arxiv.org/abs/1806.07528), Lacoste et al. 2018
* [Conditional Neural Processes](https://arxiv.org/abs/1807.01613), Garnelo et al. 2018
* [Neural Processes](https://arxiv.org/abs/1807.01622), Garnelo et al. 2018


## Representation Learning
* [Ladder variational autoencoders](http://papers.nips.cc/paper/6275-ladder-variational-autoencoders), Sønderby et al. 2016 
* [PixelVAE: A Latent Variable Model for Natural Images](https://arxiv.org/abs/1611.05013), Gulrajani et al. 2016
* [Variational Lossy Autoencoder](https://arxiv.org/abs/1611.02731), Chen et al. 2016
* [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349), Bowman et al. 2015
* [Generating Sentences by Editing Prototypes](https://arxiv.org/abs/1709.08878), Guu et al. 2017
* [The Variational Fair Autoencoder](https://arxiv.org/abs/1511.00830), Louizos et al. 2015
* [VAE with a VampPrior](https://arxiv.org/abs/1705.07120), Tomczak and Welling 2017
* [Hierarchical VampPrior Variational Fair Auto-Encoder](https://drive.google.com/file/d/1G9vfra-BEgLAQhhfguagT9m72lCVTfWP/view?usp=drive_web), Botros and Tomczak 2018
* [Neural Relational Inference for Interacting Systems](https://arxiv.org/abs/1802.04687), Kipf et al. 2018 
* [Hyperspherical Variational Auto-Encoders](https://arxiv.org/abs/1804.00891), Davidson et al. 2018

## Disentanglement in Deep Representations
* [Emergence of Invariance and Disentanglement in Deep Representations](https://arxiv.org/abs/1706.01350), Achille and Soatto 2017
* [Early Visual Concept Learning with Unsupervised Deep Learning](https://arxiv.org/abs/1606.05579), Higgins et al. 2016
* [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl), Higgins et al. 2017
* [Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942), Chen et al. 2018
* [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599), Burgess et al. 2018


## Memory Addressing as Inference

## Discrete Latent Variable

* [The concrete distribution: A continuous relaxation of discrete random variables](https://arxiv.org/abs/1611.00712), Maddison et al. 2016
* [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144), Jang et al. 2016


## Bayesian Deep Neural Networks (Variational Approaches)
* [Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks](https://arxiv.org/abs/1502.05336), Hernández-Lobato and Adams 2015
* [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142), Gal and Ghahramani 2015
* [Variational Dropout and the Local Reparameterization Trick](https://arxiv.org/abs/1506.02557), Kingma et al. 2015
* [Dropout Inference in Bayesian Neural Networks with Alpha-divergences](https://arxiv.org/abs/1703.02914), Yingzhen and Gal 2017
* [Multiplicative Normalizing Flows for Variational Bayesian Neural Networks](https://arxiv.org/abs/1703.01961), Louizos and Welling 2017
* [Bayesian Hypernetworks](https://arxiv.org/abs/1710.04759), Krueger et al. 2017
* [Deep Prior](https://arxiv.org/abs/1712.05016), Lacoste et al. 2017
* [Noisy Natural Gradient as Variational Inference](https://arxiv.org/pdf/1712.02390.pdf), Zhang et al. 2018

## Bayesian Deep Neural Networks (MCMC Approaches)

* [Bayesian Learning via Stochastic Gradient Langevin Dynamics](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf), Welling and Teh 2011
* [Bayesian Posterior Sampling via Stochastic Gradient Fisher Scoring](https://arxiv.org/pdf/1206.6380.pdf), Ahn et al. 2012
* [Stochastic Gradient Hamiltonian Monte Carlo](https://arxiv.org/pdf/1402.4102.pdf), Chen et al. 2014
* [Bayesian Sampling Using Stochastic Gradient Thermostats](http://papers.nips.cc/paper/5592-bayesian-sampling-using-stochastic-gradient-thermostats), Ding et al. 2014
* [Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks](https://arxiv.org/pdf/1512.07666.pdf)
* [Entropy-SGD: Biasing Gradient Descent Into Wide Valleys](https://arxiv.org/pdf/1611.01838.pdf), Chaudhari et al. 2017




## Deep neural networks = Gaussian Process


## SGD / Approximate Inference / PAC-Bayes
* [PAC-Bayesian Theory Meets Bayesian Inference](https://arxiv.org/abs/1605.08636), Germain et al. 2016
* [Stochastic Gradient Descent as Approximate Bayesian Inference](https://arxiv.org/abs/1704.04289), Mandt et al. 2017
* [Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks](https://arxiv.org/abs/1710.11029), Chaudhari and Soatto 2017
* [Generalization Bounds of SGLD for Non-convex Learning: Two Theoretical Viewpoints](https://arxiv.org/abs/1707.05947), Mou et al. 2017
* [Entropy-SGD optimizes the prior of a PAC-Bayes bound: Generalization properties of Entropy-SGD and data-dependent priors](https://arxiv.org/abs/1712.09376), Dziugaite and Roy 2017






