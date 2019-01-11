# Action Grammars: A Grammar-Induction Based Method for Learning Temporally-Extended Actions
## Authors: Robert Lange and Aldo Faisal | January 2019

Implementation of ICML 2019 submitted paper.


## DONE:
* [x] Create env, repo and base structure


## TODO - CODING:
* [ ] Write setup script
* [ ] Restructure Towers of Hanoi setup (State: Final MSc Project)
    * [x] Base Q-Learner
    * [x] Add logging module
    * [x] Write base Macro module
    * [x] Write function that checks executability of macro
    * [x] Check what is wrong with rollout function
    * [x] Macro Q-Learner
    * [ ] Set of k-Sequitur CFG inference
* [ ] Towers of Hanoi Experiments
    * [ ] Implement better baselines
    * [ ] Investigate transfer learning via grammars
    * [ ] Implement grammar dictionary similiar to ER buffer


## TODO - PAPER:
* [x] Time plan for ICML 2019
* [ ] Rough structure - write text/base body
* [ ] Read Pineau DRL overview paper
* [ ] Formalize DQN with macros


## Repository Structure
```
Action Grammars
+- workspace.ipynb: Main workspace notebook - Execute for replication
```

## (Basic) How to use this code
1. Clone the repo, create/activate venv, install dependencies
```
git clone https://github.com/RobertTLange/action-grammars-hrl
cd action-grammars-hrl
virtualenv -p python AG
source AG/bin/activate
pip install -r requirements.txt
```
2. Install all remaining dependencies:
```
source setup.bash
```
3. Run the main notebook:
```
jupyter notebook workspace.ipynb
```

## (Advanced) Jupyter Env Setup and Setup on AWS Virtual Machine Instance

* Create the environment, activate it and install dependencies
```
conda create --name AG python=3.6 --no-default-packages
source activate AG
pip install -r requirements.txt --quiet
```
Add ipykernel to listed env kernels, Launch notebook silent and open port
```
python -m ipykernel install --user --name AG --display-name "Python3 (AG)"
jupyter notebook --no-browser --port=8080
```
In new terminal window on local machine rewire port and listen
```
ssh -N -f -L localhost:2411:localhost:8080 MACHINE_IP_ADDRESS
```
In Browser open localhost port and start working on the notebook of choice.
If required copy paste the token/set a password.
```
localhost:2411
```

## Jupyter Env Cleanup
conda env remove -n AG
jupyter kernelspec uninstall AG




* Create SSH tunnel in local terminal

```
```

* Go to browser and enter the token when asked

```

```
