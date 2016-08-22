
<!-- MarkdownTOC -->

- [running an experiment](#running-an-experiment)

<!-- /MarkdownTOC -->

# running an experiment
```
git clone https://github.com/wulfebw/qnets.git
cd qnets
virtualenv venv
source venv/bin/activate
pip install gym numpy box2d
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
# install box2d somehow
cd tests
python test test_agent.py TestSequenceAgent.test_sequence_agent_on_advanced_mdp
```