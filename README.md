# RadDQN
**RadDQN: a Deep Q Learning-based Architecture for Finding Time-efficient Minimum Radiation Exposure Pathway**

Description:  We introduce RadDQN architecture that works on efficient reward-based function to provide time-efficient minimum radiation exposure based optimal path. Inclusion of information about position and strength of radiation source, radial distances of the agent to the radioactive sources and from destination makes the reward-based function effective to deal with multiple scenarios that significantly differs on distribution of radiation field. Further, we propose unique exploration strategies that enables the conversion of random action into a model-directed action based on the importance of future reward and progress of the training. In our article (link), we have demonstrated the performance of agent in multiple scenarios with varying number of source and its strength. Moreover, we benchmarked the predicted path with grid-based deterministic Dijkstra algorithm. Our model is found to achieve superior convergence rate and high training stability as compared to vanilla DQN.

**Link of the article (preprint) at ArXiv**
https://arxiv.org/abs/2402.00468

**Flowsheet of RadDQN architechture:**

![Real_RadDQN](https://github.com/BiswajitSadhu/RadDQN/assets/96395651/e5452241-7591-4640-a242-c72b6ba9bdc6)

**Radiation Aware Reward Structure:**

![radiation_aware_reward_function_1_1_colorbar-1](https://github.com/BiswajitSadhu/RadDQN/assets/96395651/7d2085b4-f916-4209-8cfb-56e8c2852ff1)

Radiation-aware reward function for two sources of unit radiation strength at (2,0) and (7,7). Reward \textbf{r} results from the subtraction of $\frac{n}{R_{e}}$ from $\sum_i{\frac{\Gamma S_i}{R_{s,i}^{2}}}$. In the figure, the start and exit cell are symbolized as S and E, respectively. The value of $\Gamma$ and n are taken as 1.

**Run command:**

User needs to provide the path for saving log file. The seed may be assigned using --seed option.

**python main.py --config_file configs/dqn_ef_1_s_0.yaml --logdir log/seed_3007_ef_1_s_0 --seed 3007**

**Scenario:** 

Three radioactive sources on the simulated floor. The task of the agent is to reach destination in quickest time but with exposure of minimum radiation intensity (cumulatively).

![variable_three_sources](https://github.com/BiswajitSadhu/RadDQN/assets/96395651/86321f01-53a4-45f9-9882-871e148909ca)

The above plot shows the optimum path (black dashed line) predicted by RadDQN in case of three sources (S1, S2 and S3) in simulated floor. The important diversion points within the predicted path in response to the change in radiation intensity of sources are shown as A/B/C/D. (a) Top panel: S1, S2 and S3 has equal radiation strength (5 unit) (b) bottom left panel: radiation strength of S1 source is increased by 20-fold. The trajectory has three major diversion points to minimize the radiation exposure. (c) bottom right panel: radiation strength of S1 and S2 sources are increased by 20-fold. The trajectory has four major diversion points to minimize the radiation exposure.
