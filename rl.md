# ICML Day 2 RL

### Guided Cost Learning Algorithm

An IRL about the NN cost function using human-recorded logs (multiple epoches).

Algorithm
* generate policy samples from q
* update cost using samples demos
* update q w.r.t cost (partially optimize

Ho et al ICML 16 , kim bengio arxiv 16

Simulated Wxperiments, comparing path integral IRL a& relative entropy IRL
* 2D navigation
* 2D reaching
 
### Smooth Imitation Learning

Motivation: learning/imitate camera operation (smooth for continuous action domain)

* we prefer smooth trajectories, \pi(s, a) is \gamma-lipschthe continuous in the second variable
* naive superised learning with filtration will be unstable
* with a smooth regularizer, we sacrifice a lot of accuracy
* SIMILE:Smooth IMItation LEarning

Algithm
* generate trajectory using \pi
* get smooth feedback
* tilda a_t = sigma star a_t +(1-sigma) a_t

Related approach SEARN in discrete case

### Continuous DQN with model-based acceleration

Motivation: need too many sample, need to be sample efficient
-> extent DQN for continuous action, and combined it with model-based RL
why: b/c model based ones could be very sample efficientf
related DDPG deep deterministic policy gradient, then actor-critic method
NAF Algo
* split Q into A advantage function and V value function
* solve within 1000 episodes
imagination rollouts
* run exploration plicy
* store in replay
* fit Q function

### Asynchro DRL

DQN 8 days on a GPU, want fast training on a single machine

Async Advantage Actor-Critic

### Estimating the Maximum Exppected Value through Gaussian Approximation

* The usual approach of taking the maximum of the sample means leads to large overestimates that may significantly harm the performance of the learning algorithm.
* A new estimator proposed based on a weighted average of the sample means, where the weights are computed using Gaussian approximations for the distributions of the sample means.
* 

###  Data-Efficient Off-Policy Policy Evaluation
how can we know how god a policy will be without running it, with historical data
* objective: minimize MSE E[(p(\pi)-p(\hat{\pi})^2]
* strong consistency lim_{T->\inf}MSE=0

Algorithm: IS imortant sampling
IS has strong consistent but with high variance
We want to build a AM approximate model which mimic the true world AM(pi|D)=p(pi|D)
sacrifice some consistency to reduce the variance

Proposed
* blemd IS & AM
* use IS for 4 steps and AM for the rest
* combine off-policy j-step returns
* minimize MSE return
* 
### Cumulative Prospect Theory x RL

This is an interesting interdiciplinary research of psycho and RL
* CPT captures human performance
* the feeling of losing 10k is bigger than wining 10k -> utility function
