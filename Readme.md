# Action Grammars: A Grammar-Induction Based Method for Learning Temporally-Extended Actions
## Authors: Robert Lange and Aldo Faisal | January 2019

Implementation of ICML 2019 submitted paper.


## DONE:
* [x] Create env, repo and base structure


## TODO - CODING:
* [ ] Write setup script
* [ ] Restructure Towers of Hanoi setup (State: Final MSc Project)
    * [ ] Base Q-Learner
    * [ ] Add logging module
    * [ ] Macro Q-Learner
* [ ] Towers of Hanoi Experiments
    * [ ] Implement better baselines
    * [ ] Investigate transfer learning via grammars
    * [ ] Implement grammar dictionary similiar to ER buffer


## TODO - PAPER:
* [x] Time plan for ICML 2019
* [ ] Rough structure - write text/base body
* [ ] Read Pineau DRL overview paper
* [ ] Formalize DQN


## Repository Structure
```
Action Grammars
+- workspace.ipynb: Main workspace notebook - Execute for replication
```

## How to use this code
1. Clone the repo.
```
git clone https://github.com/RobertTLange/action-grammars-hrl && cd action-grammars-hrl
```
2. Create a virtual environment (optional but recommended).
```
virtualenv -p python AG
```
Activate the env (the following command works on Linux, other operating systems might differ):
```
source AG/bin/activate
```
3a. Install all dependencies via pip:
```
pip install -r requirements.txt
```
3b. Install all remaining dependencies:
```
source setup.bash
```
4. Run the main notebook:
```
jupyter notebook workspace.ipynb
```

## Jupyter Env Setup
conda create --name AG python=3.6 --no-default-packages
source activate AG
pip install ipykernel
python -m ipykernel install --user --name AG --display-name "Python3 (AG)"
pip install jupyterlab

conda env remove -n AG
jupyter kernelspec uninstall AG
