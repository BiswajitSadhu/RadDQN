# RadDQN
RadDQN: a Deep Q Learning-based Architecture for Finding Time-efficient Minimum Radiation Exposure Pathway

Description:  We introduce RadDQN architecture that works on efficient reward-based function to provide time-efficient minimum radiation exposure based optimal path. Inclusion of information about position and strength of radiation source, radial distances of the agent to the radioactive sources and from destination makes the reward-based function effective to deal with multiple scenarios that significantly differs on distribution of radiation field. Further, we propose unique exploration strategies that enables the conversion of random action into a model-directed action based on the importance of future reward and progress of the training. In our article (link), we have demonstrated the performance of agent in multiple scenarios with varying number of source and its strength. Moreover, we benchmarked the predicted path with grid-based deterministic Dijkstra algorithm. Our model is found to achieve superior convergence rate and high training stability as compared to vanilla DQN.

Flowsheet of RadDQN architechture:

![Real_RadDQN](https://github.com/BiswajitSadhu/RadDQN/assets/96395651/e5452241-7591-4640-a242-c72b6ba9bdc6)

Run command:

User needs to provide the path for saving log file. The seed may be assigned using --seed option.

python main.py --config_file configs/dqn_ef_1_s_0.yaml --logdir log/seed_3007_ef_1_s_0 --seed 3007

