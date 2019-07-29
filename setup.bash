################################################################################
# SETUP OF EXPERIMENTS
################################################################################
# Clone & install the Towers of Hanoi Environment
git clone https://github.com/RobertTLange/gym-hanoi
cd gym_hanoi
python setup.py install
# Create empty results folder
mkdir results
# Setup the Grammar induction pipeline
cd ..
cd grammars
# Clone Sequitur and Install - Forked from craignm @29/07/19
git clone https://github.com/RobertTLange/sequitur
cd sequitur
make
