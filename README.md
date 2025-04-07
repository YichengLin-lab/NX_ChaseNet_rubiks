**NX_ChaseNet_rubiks**

The implementation and model weights for paper https://arxiv.org/pdf/2411.19583v1

NX_rubiks uses a model termed ChaseNet to estimate the cost between states of cube and uses PPO to train actor network based on the feedback from ChaseNet. Different from most of other trials using RL to solve rubiks cube, NX_rubiks doesn't rely on sampling near-solved states. For more implementation details please refer to the paper.

Download the files from https://drive.google.com/drive/folders/1suxVmm7FBZHN_9IEa_HvuLaV3HM0W1qX?usp=sharing and put the "rubiks_cube_222_states_FRU.pickle" in the current directory (the same directory as main.py). Then put the following files in a folder manually created named "models_saved":

nx_network_warmup.pt: The ChaseNet-attention after the warmup phase (As described in the paper)

nx_network_during_ppo_last.pt: The ChaseNet-attention after the training phase.

ppo_actor_last.pt: The actor that takes in the observations and directly take actions, denoted as : 0:F, 1:R, 2:U

**Warning**

The three .pt files downloaded are trained model instances that we strongly recommend you backup somewhere. You can test these models safely without modification. However if you try to reproduce the training process by running main.py directly, the new model instances will cover these three files at each step of the training. If the training process is not completed, the newly saved models will lead to poor performance.

**Getting Started**

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
python test_nx_model.py # Test the accuracy of chasenet
python test_actor.py # Test the recover rate of the actor
```

You can visualize how the actor solved the pocket cube in the expanded view:

```shell
python test_actor_visual.py
```

Enjoy!
