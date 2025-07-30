# Paul's Research Log
### July 28th, 2025
##### Implementing basic DQN algorithm

### July 30th, 2025
Performed a 50-run sweep of an MLP DQN agent learning the CartPole-v1 environment. The sweep, along with the config,
can be found [here](https://wandb.ai/kan_rl/Buffer-test/sweeps/4dtpqxy6)

Only 2 runs seemed to both learn and maintain knowledge of how to perform the task, namely
[Devoted Sweep](https://wandb.ai/kan_rl/Buffer-test/runs/auw3m64a) and [Magic Sweep](https://wandb.ai/kan_rl/Buffer-test/runs/7vu8epy1).
Magic did much better, maintaining an average return of about 90 and episode lengths exceeding 200. Clearly, it was able to balance the pole.
