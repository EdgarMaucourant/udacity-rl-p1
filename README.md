# Project Details
This repository hosts the submited project of Edgar Maucourant for the grading project of the Value-Based Methods module of the Deep Reinforcement Learning Nanodegree at Udacity.

It uses Unity ML-Agents and Python scripts to train an agent to play the Bananas games where the player is expected to navigate an environment seeking for yellow bananas while avoiding blue ones. Each bananas counts for 1 point.

Here are the details of the environment

| Type				| Value		|
|-------------------|-----------|
| Action Space      |  4        |
| Observation Shape |  (37,)    |
| Solving score     |  13       | 

Here is an example video of an agent trained for 2000 iterations able to solve the game with 17 bananas collected:

<video width="1290" height="762" controls>
  <source src="P1_bananas.mp4" type="video/mp4">
</video>

Please follow the instructions below to train your agents using this repo. Also please look into the [Report](Report.md) file to get more info about how the code is structured and how the model behave under training.

# Getting Started

Before training your model, you need to download and create some elements.

*Note:*  this repo assume that your are running the code on a Windows machine (the Unity game is only provided for Windows) however adapting it to run on Mac or Linux should only require to update the path the the game executable, this has not been tested though.

## Create a Conda env
1. To be able to run the training on a GPU install Cuda 11.6 from (https://developer.nvidia.com/cuda-11-6-2-download-archive)

2. Create (and activate) a new environment with Python 3.7.

```On a terminal
conda create --name drlnd python=3.7 
conda activate drlnd
```
	
3. Install the dependency (only tested on Windows, but should work on other env as well):
```bash
git clone https://github.com/EdgarMaucourant/udacity-rl-p1
pip install .
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment. 
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

## Instructions to train the agent

To train the agent, please follow the instruction under the section *4. It's Your Turn!* of the Navigation Jupyter Notebook.

1. From a new terminal, open the notebook

```
jupyter notebook Navigation.ipynb
```

2. Scroll to the section *4. It's Your Turn!* and run the cell defining the function "dqn". This function is used to train the agent using the hyperparameters provided. Note that in our cases we used the default parameters for Number of episodes (2000), max steps (1000), and epsilon (start=1.0, end=0.1, and decay-0.995).

3. Run the next cell to import the required dependencies, and create a new environment based on the Bananas game (note that this is where you want to update the reference to the executable if you don't run on Windows). 

This cell also create the Agent to be trained, that agent is based on the Deep Q-Learning Algorithm and expect the state size and action size as input (plus a seed for randomizing the initialization). For more details about this agent please look at the [Report](Report.md).

4. Run the next cell to start the training. After some time (depending on your machine, mine took about 10 minutes), your model will be trained and the scores over iterations will be plotted. Note that while training you should see the game running (on windows at least) and the score increasing. If after 500 iterations the score did not increase you might want to review the parameters you provided to the dqn function.

*Note:* the last parameter passed to the dqn function in that cell "13" is the average score to obtain over 100 attempts to succeed the training. It is based on the requirement of the project.

## Instructions to see the agent playing

The last cell in the Jupyter notebook shows how to run one episode with a model trained (the pre-trained weights are provided), if you run the cells (after having imported the dependency and created the env, see step 3 and 4 above) you should be able to see the game played by the agent (if you run this code locally on a Windows machine). See how much you agent can get! The videos at the top of this document shows the agent running with the pre-trained weights provided achieving a score of 17.
