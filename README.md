# ICML Workshop Talk & AMA Notes

## Day 1: What is deep learning in the small data regime?

Under this broad theme, more specific questions may include “Does unsupervised learning have a central role in this?”, “What else is essential when dealing with high sample complexity but low resource?” and even more radical ones such as “Should deep learning be replaced with an alternative in the small data regime?”

* Deep Learning Workshop https://sites.google.com/site/dlworkshop16/

Deep learning is a fast-growing field of machine learning concerned with the study and design of computer algorithms for learning good representations of data, at multiple levels of abstraction. There has been rapid progress in this area in recent years, both in terms of methods and in terms of applications, which are attracting the major IT companies as well as major research labs. Many challenges remain, however, in aspects like large sample complexity of deep learning approaches, generative modeling, learning representations for reinforcement learning and symbolic reasoning, modeling of temporal data with long-term dependencies, efficient Bayesian inference for deep learning and multi-modal data and models. This workshop aims at tackling two major challenges in deep learning, which are unsupervised learning in the regime of small data, and simulation-based learning and its transferability to the real world, by bringing together researchers in the field of deep learning.

* Abstraction in Reinforcement Learning http://rlabstraction2016.wix.com/icml

Many real-world domains can be modeled using some form of abstraction. An abstraction is an important tool that enables an agent to focus less on the lower level details of a task and more on solving the task at hand. Temporal abstraction (i.e., options or skills) as well as spatial abstraction (i.e., state space representation) are two important examples. The goal of this workshop is to provide a forum to discuss the current challenges in designing as well as learning abstractions in real-world Reinforcement Learning (RL).

* Machine Learning for Music Discovery https://sites.google.com/site/ml4md2016/

The ever-increasing size and accessibility of vast music libraries has created a demand more than ever for machine learning systems that are capable of understanding and organizing this complex data. Collaborative filtering provides excellent music recommendations when the necessary user data is available, but these approaches also suffer heavily from the cold-start problem. Furthermore, defining musical similarity directly is extremely challenging as myriad features play some role (e.g., cultural, emotional, timbral, rhythmic). The topics discussed will span a variety of music recommender systems challenges including cross-cultural recommendation, content-based audio processing and representation learning, automatic music tagging, and evaluation.

**AMAs**

1. As much of the success in working with small sample sizes involves transfer learning in some way, e.g. finetuning or feature extraction using a network pretrained on ImageNet or something similar, it seems this process is inherently suboptimal since both the original architecture of the network (whether AlexNet, VGG, or whatever) and the ImageNet dataset weren't made with transfer learning in mind.
With that in mind, are we going to see general-purpose networks designed specifically for transfer learning? Is unsupervised pre-training going to make a comeback? More broadly speaking, is transfer learning even seen as the solution to small sample size problems, or is there something better in the works?

2. One obvious (in principle) approach to tackle the small data regime is to use strong priors. A prior may be a) hand-coded and domain-specific, b) hand-coded and domain-neutral (e.g. a simplicity prior a la Solomonoff induction) c) learned and domain-specific (transfer learning, empirical Bayes), d) (for completeness) learned and domain-neutral (learned from what? I've never actually seen this in practice).
Things like dropout, stochastic depth, etc. kinda fall in (b), since they could be considered implicit domain-neutral priors, and empirically they improve generalization in the the big data regime, but so far they haven't been sufficient to tackle the small data regime.
Recent works by DeepMind (Santoro et al. 2016, Vinyals et al. 2016) achieved significant success by taking approach (c): they train a network on many similar one-shot problems, each treated as a separate episode and then they apply it to novel problems without changing its weights, so that the network weights and act like a kind of learned prior and its run-time state is used to store representations of the current episode and compute the posterior (roughly, since these models are not strictly Bayesian).
It should be noted that these approaches are not strictly "small data" since they still require large labeled datasets to be trained, although once they are trained they generalize well after seeing only few examples of a novel classification class, assuming that they aren't too dissimilar from the ones in the large training set.
Do you think that the other approaches (a), (b) and possibly (d) have any merit? Could they be combined with deep learning?
Intuitively, it would seem that probabilistic programming is better suited to using explicit priors, but perhaps there could be a way to use them with deep learning as well. Any ideas?

3. Another follow-up: Given a sufficient number of such systematically biased datasets and corresponding unbiased supervised labels, are there deep learning architectures are capable and/or efficient at removing these "batch effects" (aka automated normalization)?
EDIT for clarification: A crude linear way of removing batch effects might be retaining the residual deviance + the deviance explained by X from regressing Y ~ B + X, where Y is the dependent variable, X are the independents, and B is the batch label. Might deep learning be capable of performing such normalization automatically? (e.g. automatically inferring latent batch labels from X and regressing them out with something like Residual Networks)?
You could probably do this with adversarial training - train a secondary predictor to try to estimate which batch a given sample came from given only the hidden layer activations of your primary network, then add a term to the loss function of the primary network which encourages it to only retain information in the intermediate layers that make the secondary network's job hard. https://www.robots.ox.ac.uk/~vgg/rg/papers/Tzeng_ICCV2015.pdf is pretty much this idea.

## Day 2: What does simulation-based learning bring to the table?

Under this broad theme, more specific questions may include “How transferrable is the knowledge learned from a simulation to the real world?", "Are simulated environments the way to achieve machine intelligence?" and "How important is it for agents to simulate the world in their minds?” Even more radical ones will also be discussed, such as “Does simulation-based learning have anything to do in making progress toward artificial intelligence?”

* Online advertising systems https://sites.google.com/site/admlsystemsworkshop/

Online advertising is a multi-billion dollar industry driven by the confluence of machine learning, optimization, control systems, auction algorithms, econometrics and software engineering. The goal of this workshop is to discuss how machine learning systems operate within the context of an advertising system.

**AMAs**

1. What is the best way to augment simulated data with real world data?
Said another way, is it possible to take a hybrid approach wherein the simulation is primarily based on real world data but we can use simulation or physical models to perturb or extrapolate rare situations or situations not yet seen in the data, creating a more complete manifold for the network to learn?
I think combining the two into a single control+perception loop will offer great benefits. Simulation allows us to perform directed exploration of state space. Thus a simulation module helps the learning system to find good solutions quickly by reducing sample complexity. The simulation module will also benefit since it can adapt by understanding the limitations of the learner.
Concretely, learning an end-to-end system from observations to actions is going to produce better results than separately trained vision and trajectory optimization systems glued together. In fact, it is proven mathematically that for any non-linear system, de-coupling the two modules leads to loss in performance. (For linear systems, Kalman Filter + LQR is optimal)

2. I think combining the two into a single control+perception loop will offer great benefits. Simulation allows us to perform directed exploration of state space. Thus a simulation module helps the learning system to find good solutions quickly by reducing sample complexity. The simulation module will also benefit since it can adapt by understanding the limitations of the learner.
Concretely, learning an end-to-end system from observations to actions is going to produce better results than separately trained vision and trajectory optimization systems glued together. In fact, it is proven mathematically that for any non-linear system, de-coupling the two modules leads to loss in performance. (For linear systems, Kalman Filter + LQR is optimal)

3. Problems like Atari game simulation live completely in the virtual world, where training and testing are in the same environment (~ALE). In physical control tasks like robotics, an additional question is: "is my simulation model a correct reflection of the environment?" It's clear that training from scratch on physical systems is hopeless due to costs of badly chosen actions (Sergey Levine describes this would lead to bipedal falling instead of walking), and hence model or simulation based learning is needed. How well do these results transfer to the physical world, and secondly how can the learning community help in getting better models/simulators of the environment - a pertinent question since the quality of model directly affects the quality of solution in practice.
A closely related community which has been looking at this issue for long is optimal control, especially intelligent control (https://en.wikipedia.org/wiki/Intelligent_control) which has looked at using neural networks as non-linear controllers; and system identification. Are there cross fertilization of results between communities to avoid reinventing the wheel. Some folks like Emo Todorov, Russ Tedrake, and Stefan Schaal could potentially act as a bridge between the communities.

4. I'm currently working on creating a self-driving car in a simulation, namely GTAV, and care deeply about this so thank you for asking!
Given that deep convnets can recognize drawings and stuffed animals versions of the real-world objects it has been trained on with datasets like ImageNet (Jeff Dean's presentations notoriously include Google+ recognizing a Yoda stuffed animal), would a mixed training approach work where the training data had examples from both simulation and real-world environments? Also would a different mix in training vs test data make sense, say testing more on real-world to discover regularization or other techniques that led to better performance?
Anecdotally, I've noticed that nets pre-trained for ImageNet classification work quite well to speed up learning in GTAV's environment on non-classification tasks like regression and RL. Also, I've seen several examples of nets trained for classification work well in regression tasks like detection, segmentation, etc... so the representation seems to transfer not only to other input types, but different output types as well. Unfortunately, RL tasks can not as easily use the mixed approach, since interaction with the environment generates future training data. However, could one use something like DQN's experience replay to mix simulated and real experiences in training? Also, what other approaches could be used to transfer between simulated and real-world environments in an RL context? Finally, is there value in iterating the process, i.e. using nets transferred into real-world environments as the basis for the next generation of nets trained in simulation. If so, is there some limit to how many times a net can be transferred?
Edit: This paper demonstrates the ability of a convnet to transfer well from a video-game to a real-world car without any extra training: http://deepdriving.cs.princeton.edu/
We are trying to do something similar with convnets where we want to do continual learning between controlled environments and "real world domains" like imagenet, this pose the huge problem of catastrofic interference: once you begin adapting your weights for the new environment the network forgets the ability to perform in the virtual environment.
We face the problem with a mix of replay (potentially substitutable with an environment conditioned generative process) and knowledge distillation using a dual architecture, similar to the policy distillation framework https://arxiv.org/abs/1606.02355
Something that popped out in ICLR around the posters was intermixing in DQN visual tasks like those from the perceptual learning field to learn to "watch" things.
