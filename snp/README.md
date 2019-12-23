# Sequential Neural Processes

This repository includes the implementation of the paper, [Sequential Neural Processes (Gautam Singh and Jaesik Yoon et al., NeurIPS 2019)](https://arxiv.org/abs/1906.10264). This directory contains the code for running the 1D regression task described in the paper.

The code for Temporal GQN is provided in the other directory `temporal-gqn`.

This code is based on [Deepmind's Neural Processes](https://github.com/deepmind/neural-processes).

One may check or install the libraries used in our experiments as follows.
```
pip install -r requirements.txt
```

## Quick Run
The instructions for running the experiments described in the paper are as followed.

* Examples of SNP run scripts
```
# case a running scripts
python main.py --MODEL_TYPE=NP --dataset=gp --case=1 --LEN_SEQ=20 --LEN_GIVEN=10 --LEN_GEN=0 --MAX_CONTEXT_POINTS=50 --log_folder=logs_gp_a --batch_size=16 --l1_min=0.7 --l1_max=1.2 --sigma_min=1.0 --sigma_max=1.6
python main.py --MODEL_TYPE=SNP --dataset=gp --case=1 --LEN_SEQ=20 --LEN_GIVEN=10 --LEN_GEN=0 --MAX_CONTEXT_POINTS=50 --log_folder=logs_gp_a --batch_size=16 --l1_min=0.7 --l1_max=1.2 --sigma_min=1.0 --sigma_max=1.6

# case b running scripts
pyton main.py --MODEL_TYPE=NP --dataset=gp --case=2 --LEN_SEQ=20 --LEN_GIVEN=10 --LEN_GEN=0 --MAX_CONTEXT_POINTS=50 --log_folder=logs_gp_b --batch_size=16 --l1_min=0.7 --l1_max=1.2 --sigma_min=1.0 --sigma_max=1.6
python main.py --MODEL_TYPE=SNP --dataset=gp --case=2 --LEN_SEQ=20 --LEN_GIVEN=10 --LEN_GEN=0 --MAX_CONTEXT_POINTS=50 --log_folder=logs_gp_b --batch_size=16 --l1_min=0.7 --l1_max=1.2 --sigma_min=1.0 --sigma_max=1.6

# case c running scripts
python main.py --MODEL_TYPE=NP --dataset=gp --case=3  --LEN_SEQ=50 --LEN_GIVEN=45 --LEN_GEN=0 --MAX_CONTEXT_POINTS=10 --log_folder=logs_gp_c --batch_size=16 --l1_min=1.2 --l1_max=1.9 --sigma_min=1.6 --sigma_max=3.1
python main.py --MODEL_TYPE=SNP --dataset=gp --case=3  --LEN_SEQ=50 --LEN_GIVEN=45 --LEN_GEN=0 --MAX_CONTEXT_POINTS=10 --log_folder=logs_gp_c --batch_size=16 --l1_min=1.2 --l1_max=1.9 --sigma_min=1.6 --sigma_max=3.1
```

One may obtain detailed results using Tensorboard.

## Contact
Any feedback is welcome! Please open an issue on this repository or send email to Jaesik Yoon (jaesik.yoon.kr@gmail.com), Gautam Singh (singh.gautam@rutgers.edu) or Sungjin Ahn (sungjin.ahn@rutgers.edu).

