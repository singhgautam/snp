# Temporal Generative Query Networks

## Requirements
- Python 3
- tensorboardX>=1.4
- torch>=0.4.1
- torchvision>=0.2.1
- opencv-python>=4.1.2
- tensorboard (For visualization of running logs)

## How to Train
With `temporal-gqn` as the working directory, execute the script `scripts/perform_training.sh`.

## Computing Requirements
The experiments in the paper were run on **4x GPUs with roughly 8 GB memory**.

## Dataset
The code provided here generates the 2D color shapes dataset episodes on the fly at training/test time.


## Usage

| File                              | Usage                                      |
| --------------------------------- | ------------------------------------------ |
| `scripts/perform_training.sh`     | Launch training.                           |
| `scripts/perform_evaluation.sh`   | Launch evaluation on a trained model file. |
| `train.py`                        | Main training code.                        |
| `evaluate.py`                     | Main evaluation code.                      |
| `datasets/colorshapes_dataset.py` | 2D Color Shapes dataset                    |
| `datasets/context_curriculum.py`  | Context curriculum per episode             |
| `models/tgqn_pd.py`               | Temporal GQN                               |
| `models/gqn.py`                   | CGQN baseline (based on Kumar et al.)      |

## References
- Kumar, Ananya, et al. "Consistent generative query networks." arXiv preprint arXiv:1807.02033 (2018).
