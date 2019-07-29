# Action Grammars: A Grammar-Induction Based Method for Learning Temporally-Extended Actions
## Authors: Robert Lange & Aldo Faisal | July 2019

Implementation of CCN 2019 accepted paper. For the full paper click here. In this work we combine Hierarchical Reinforcement Learning and Grammar induction algorithms in order to define temporal abstractions. These can then be used to alter the action space of the Reinforcement Learning agent.

This work originated during a MSc project at the FaisalLab at Imperial College London.

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
2. Install all remaining dependencies (Hanoi Env, Sequitur):
```
source setup.bash
```
3. Run the experiments:
```
bash run_experiments.sh $toh
bash run_experiments.sh $gridworld
```
4. Visualize the results:
```
jupyter notebook visualize_results.ipynb
```

## Get in touch!

There are many possible avenues that I want to explore in the following months. Feel free to contact me in case of any questions!

August 2019,

Robert Tjarko Lange

## ToDo-List
* [ ] Parallelize over runs
* [ ] Compress learning functions - functional/object oriented programming
* [ ] Add random seed & run experiments over longer time again!
* [ ] Actually look into the grammars
* [ ] Fix online towers grammar inference
* [ ] Get Lexis rolling
* [ ] Better documentation - Comment code and provide better repo structure
