**NX_ChaseNet_rubiks**

The implementation and model weights for paper https://arxiv.org/pdf/2411.19583v1

NX_rubiks uses a model termed ChaseNet to estimate the cost between states of cube and uses PPO to train actor network based on the feedback from ChaseNet. Different from most of other trials using RL to solve rubiks cube, NX_rubiks doesn't rely on sampling near-solved states. For more implementation details please refer to the paper.

Download the files from https://drive.google.com/drive/folders/1suxVmm7FBZHN_9IEa_HvuLaV3HM0W1qX?usp=sharing and put them in a folder manually created named "models_saved".

Inside folder "models_saved" are weights (model instances) of the trained models:

nx_network_warmup.pt: The ChaseNet-attention after the warmup phase (As described in the paper)

nx_network_during_ppo_last.pt: The ChaseNet-attention after the training phase.

ppo_actor_last.pt: The actor that takes in the observations and directly take actions, denoted as : 0:F, 1:R, 2:U

Getting Started

Simply run the following to install the dependencies:

```shell
pip install -r requirements.txt
```

The main.py contains everything from warmup phase to training phase to replicate the paper's results:

```shell
python main.py
```

You can test the recover rate of the actor and the accuracy of ChaseNet saved. 

```shell
python test_nx_model.py # Test the chasenet
python test_actor.py # Test the recover rate
```

You can visualize how the actor solved the pocket cube in the expanded view:

```shell
python test_actor_visual.py
```

Enjoy!