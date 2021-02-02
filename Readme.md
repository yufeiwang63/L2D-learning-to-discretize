
# Learning to Discretize: Solving 1D Scalar Conservation Laws via Deep Reinforcement Learning

This is the code for the paper "Learning to Discretize: Solving 1D Scalar Conservation Laws via Deep Reinforcement Learning", published at Communication in Computational Physics (CiCP) 2020. [[Arxiv](https://arxiv.org/abs/1905.11079)]. 
Authors: Yufei Wang*, Ziju Shen*, Zichao Long, Bin Dong (* indicates equal contribution)


## Instructions
### Preparation
1. Create conda environment  
```
conda env create -f environment.yml
```  

2. Activate the conda environment  
```
source prepare.sh
```  

### Training
1. Generate training datasets
```
python scripts/generate_solutions.py --prefix [PREFIX] --eta [ETA] --num [NUM]
```
to generate and store solutions to some randomly generated initial conditions. 
The solution files will be stored at `data/local/solutions/prefix`

2. Perform training
```
python launchers/launch_td3.py
```
This will train a TD3 agent on the pre-generated solutions of different initial conditions in step 1.  
Remember to change the `solution_data_path` variable to point to the correct path of the generated solutions in step 1.  
You can tune the parameters in `launch_td3.py`. 

The training logs will be stored at `data/local/exp_prefix`. `exp_prefix` is specified in `launch_td3.py`.  
To view the training logs, you can use `python viskit/frontend.py data/local/exp_prefix`.  
To make videos of the trained models, use `scripts/make_videos.py`  
To make error tables of the trained models, use `scripts/test_trained_model_mesh.py` or   `scripts/test_trained_model_forcing.py` or `scripts/test_trained_model_viscous.py`




