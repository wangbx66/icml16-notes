# ICML 16 Notes on Reinforcement Learning

This year's ICML features a one-and-a-half day session on reinforcement learning, which generally talks about the recent advantage of DQN, iRL, and policy optimization and evaluation. This note will also cover whatever related part of works in online learning as well as deep learning. Organized chronological.

## Day 1 RL Track

### Why Most Decisions Are Easy in Tetris

Inspecting Tetris, it's found that most individual problems encountered are easy in the following sense: One can choose well among the available actions without knowing an evaluation function that scores well in the game. Furthermore, there is an intuitive feature that typically eliminates a substantial number of actions. 

Borrowing concepts from the *comparison* problem, the underlying reasons are three types of regularities in the domain, namely, simple dominance, cumulative dominance, and noncompensation. Viewing Tetris as a series of comparison problems, the agent is able to make the decision using the tress regularities, even the actual evaluation function unknown.

The work utilizes 8 features previously proposed in Trtris frameworks. The features are assembles in an order per their corresponding linear weights in that work, to form the set of elements to be compared. The elimination shows a 70% - 90% removal of actions, while keeping a competitive overall strength of play.

* Trivia: Finding the optimal placement of teriminos is NP-complete.

### Opponent Modeling in Deep Reinforcement Learning

The work is driven by the previous studies of poker, where the agent can exploit suboptimal players if it can predict their bad moves. The goal is usually a challenging one because of strategies’ complex interaction and the non-stationary nature.

The proposed DRON (O for opponent)
* the gating part of the network acts as the original DQN, which outputs the probability distribution over the feasible actions
* the hidden value of the gating component is also used to predict the action of all other agents, in a supervised manner
* meanwhile, the expert component trys to predict the Q values of the opponents, sharing the same network until the hidden vector
* the final Q value is calculated using each Q value of the opponent, as well as the gating one.
* in the experiment, to show the strength of the opponent modeling, the author design the opponent to be either extremely offensive, or extremely defensive
* open source on [github](https://github.com/hhexiy/opponent)

### Control of Memory, Active Perception, and Action in Minecraft

The work proposes feedback recurrent memory Q-network (FRMQN), which feature the ability to retrive a prior observation from memory depends on the current context. To do this, the features are mapped linearly into a memory unit. The reading operation is based on soft attention. In this way, the recurrent input of the hidden layer is the concatenation of both the observation and the retrived memory. The hidden layer as well the memory, are finally used to calculate the Q value.

* the work is driven by the fact in Minecraft that, in order to achieve a high score, the model have to make decisions based on some early observations.

### Dueling Network Architectures for Deep Reinforcement Learning

Besides the state value function, the dueling network models the state-dependent action advantage function as well. It leads to better policy evaluation win the presense of many similar-value actions.

The algorithm is motivated by DDQN, where the action chosen in the loss function is the best predicted action based on current Q function instead of the target network. Considering Q=V+A where E[A]=0, it constructs two separate networks from the top layer of the convolution representation, to predict A for each action and V. Apply the following to identify V and A from Q,

* Q=V+A-\bar{A}; 
* replace \bar{A} with \softmax{A} should yield similar performance, but more complexity

Trivias:
* well written and enjoyable to read
* new state-of-the-art on Atari 2600

### C3 Bandit

This work extend the combinatorial cascading bandit problem to its contextual version, which makes sense in practical cases such as search and recommendation. With extended UCB it achieves guranteed squared root times logarithmatic regret.

Code available on [github](https://github.com/wangbx66/contextual-bandit).

## Day 2 RL Track

### Complex Backup Strategies in MCTS

The paper introduces back-up in RL, to UCT. During the search progress, the Q value is updated using an interpolation between the current cumulative return and the existing Q-value of that state action.

* The paper leads to empirical results, instead of theoratical ones.

### Doubly Robust Off-policy Value Evaluation

This paper is talking about the robust version of off-policy policy evaluation. The problem is extremely important where the new policies evolved can only be examined on the log data instead of a real-time interaction. Applying that to bandit problems, it gurantees to be unbiased and relative lower variance.

The previous approaches are based on the idea of importance sampling, which corrects the mismatch between the distributions induced by the target policy and by the behavior policy. Such approaches have the salient properties of being
unbiased and independent of the size of the problem’s state space, but its variance can be too large for the method to be useful when the horizon is long. Apply DR on that is able to reduce the variance, which is guranteed theoratically.

###  Model-Free Trajectory Optimization

A new algorithm as its title says. It gurantees some theoratical merits. 

### Model-Free Imitation Learning via PG

* This is an interesting work yielding both theoratical and empirical results, also related to the user imitation process in the software workflow designs

This is a direct application of policy gradient on policy imitation, finding a parameterized stochastic policy mimicking expert demonstrations. It is supposed to replace behavioral cloning which suffers from cascading errors.

Minimizing the superior gap between the policy performance and the expert performance on a set of pre-defined cost function, apprenticeship learning is able to handle some case with low-complexity of the state space, while it generally goes a iteration of reinforcement learning, which is extremely computational expensive. Inspired by LPAL, which is an AL solution based on game theory, the paper proposes to optimize gap directly over parameterized stochastic policies instead of state-action visitation distributions.

To do this, consider the fact that the policy gradient is the gradient, w.r.t. policy parameter, of the policy performance, given the cost function which maximizes the gap. The optimization is achieved by iteratively finding the cost function minima and the policy maxima. Details for that on section 4.1 onwards.

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

### Cumulative Prospect Theory x RL

This is an interesting interdiciplinary research of psycho and RL
* CPT captures human performance
* the feeling of losing 10k is bigger than wining 10k -> utility function

## ICML Day 3 MISC

### approx kemeny consensus

* rank aggregation: [kemeny](https://en.wikipedia.org/wiki/Kemeny%E2%80%93Young_method) consensus is np-hard, we want a proximity, but fast approach
* this work: given a ranking, we compute how close it is to the kemeny consensus
* experiment on sushi ranking

### rank breaking for rank aggregation

* rank breaking: break each person's ordinal preference into pairwise comparisons
* treat the paired comparisons outcomes unequally, depending on the topology of the collected data
* *best* rank breaking estimator

### neural autoregressive for CF

* recently RBM for CF
* recent work NADE. CF-NADE trys to find the conditional distribution using a network structure; 
* use random order instead of timestamp
* RBF+NADE: share parameter, and each time for one user
* scalability: fatorize the matrix into the product of two low-rank matrices
* Movielens 1M, user-based CF

###  Scalable Gradient-Based Tuning of Continuous Regularization Hyperparameters

* Learns, using gradient, the regularization factor :p

### Electricity Grid Management via RL

* formulate as MDP
* policy approx and value approx/ typical policy-value pair
* existing/classical approach + toy dataset, but sounds interesting

### forecastICU

* predict, using the sequencial data of the patient, the possibility that the patient needs an ICU treatmeant
* pretty practical work: predict the case 3 hours before it happens.

### Learning to Filter with Predictive State Inference Machines



### population level diffusion modeling via RNN

* what we observe is the individual behavior.
* mean result: what is the natural lost of this problem
* The model is based on specifying a potential function and a recurrence relationship. The method uses RNN to capture the recurrence
* only synthetic experiment
 
### Fast Parameter Inference in Nonlinear Dynamical Systems using Iterative Gradient Matching 

## ICML Day 4 RL Workshop

### toward learning representative

* motivated: create AI that helps agents to learn
* interacting machine learning problem - if you have a good teacher, the teach lern to improve its own teaching skills
* abstraction can help speed learning
* abstraction learning v.s. efficient exploration
* options (mini policies)/ macro-actions
* option discovery? 
* sample complexity - mistakes we make

algorithm
* efficient exploration by representing uncertainty over model & parameter values
* adapt representation based on data
* reduce computation
