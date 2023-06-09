# Official implementation of Neuro Symbolic Continual Learning: Knowledge, Reasoning Shortcuts and Concept Rehearsal

<p align="center">
<img src="https://github.com/ema-marconato/NeSy-CL/assets/28188499/62bf1796-1300-40bd-96a7-e39f86faf022" width=30% height=30%>
</p>


It contains a [mammuth](https://github.com/aimagelab/mammoth) implementation for our [paper](https://arxiv.org/pdf/2302.01242.pdf) accepted at ICML 2023.  If you find this code useful in your research, please consider citing:
    
    @InProceedings{marconato2023neuro,
      title={Neuro Symbolic Continual Learning: Knowledge, Reasoning Shortcuts and Concept Rehearsal}, 
      author={Emanuele Marconato and Gianpaolo Bontempo and Elisa Ficarra and Simone Calderara and Andrea Passerini and Stefano Teso},
      year={2023},
	booktitle = {Proceedings of the 40th International Conference on Machine Learning~(ICML)},
	series = {Proceedings of Machine Learning Research}




# Abstract

We introduce Neuro-Symbolic Continual Learning, where a model has to solve a sequence of neuro-symbolic tasks, that is, it has to map sub-symbolic inputs to high-level concepts and compute predictions by reasoning consistently with prior knowledge. Our key observation is that neurosymbolic tasks, although different, often share concepts whose semantics remains stable over time.Traditional approaches fall short: existing continual strategies ignore knowledge altogether, while stock neuro-symbolic architectures suffer from catastrophic forgetting. We show that leveraging prior knowledge by combining neuro symbolic architectures with continual strategies does help avoid catastrophic forgetting, but also that doing so can yield models affected by reasoning shortcuts. These undermine the semantics of the acquired concepts, even when detailed prior knowledge is provided upfront and inference is exact, and in turn continual performance. To overcome these issues, we introduce COOL, a COncept-level cOntinual Learning strategy tailored for neuro-symbolic continual problems that acquires high-quality concepts and remembers them over time. Our experiments on three novel benchmarks highlights how COOL attains sustained high performance on neuro-symbolic continual learning tasks in which other strategies fail.



## SETUP

To run the code install all dependencies with pip:
`pip install -r requirements.txt`

To test on CLEEVR, extract all files from `/clevr_rccn_pretrained_1.zip` and `/clevr_ood_test_rcnn_pretrained_1.zip` in '/data'.

## RUNS
Launch the training through `start.py` by selecting the flags for:

1- `--version` ("cbm" or "nesy");

2- `--model` (the strategy for CL, e.g. "cool");

3- `--dataset` to select the benchmark to test the model;

4- `--lr`;

5- `--batch_size`;

6- `--n_epochs` (epochs per task);

7- `--buffer_size` (if needed);

8- `--minibatch_size` (if needed);

9- strategy hyperparameters.

We included a bash script with custom configuration. It can be tested by running:
`bash run_start.sh`

## ADDITIONAL TOOLS

We included the library `submitit`, to launch jobs on slurm partitions. In that case, we included a set of jobs with best hyperparameters for each strategy in `/experiments.py`.

`wandb` is activated by passing the name of your wandb project through the flag `--wandb`.


# Reasoning Shortcuts

<p align="center">
<img src="https://github.com/ema-marconato/NeSy-CL/assets/28188499/f6c5fec1-ff97-4336-8fa3-6a8edc522eda" width=50% height=50%>
</p>

