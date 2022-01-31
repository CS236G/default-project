# CS236G Default Project
This repository contains Stanford CS236G default final project starter code. The baseline model is a modified version of [Point Cloud GAN](https://github.com/chunliangli/Point-Cloud-GAN) (ICLR'19 Workshop).

![Samples](https://user-images.githubusercontent.com/50810315/151111956-00f3fd73-5364-40bd-9ccd-d017062649af.png)

## Installation
* Set up and activate conda environment.

```shell
conda env create -f environment.yml
conda activate cs236g
```

* Compile CUDA extensions.

```shell
sh scripts/install.sh
```

* Download ShapeNet dataset and trained checkpoints.

```shell
sh scripts/download.sh
```

## Training
You can train using `train.py` or provided scripts.

```shell
# Train using CLI
python train.py --name NAME
# Train using provided settings
sh scripts/train_shapenet_airplane.sh
```

## Testing
You can evaluate checkpointed models using `test.py` or provided scripts.

```shell
# Test user specified checkpoint using CLI
python test.py --ckpt_path CKPT_PATH
# Test provided checkpoints
sh scripts/test_shapenet_airplane.sh
```

## Logging
Follow terminal instructions during the initial run to setup [Weight and Biases](https://wandb.ai) logging.
If you do not want to use Weight and Biases, you can turn it off using:

```shell
wandb offline
```

## Submitting
Generate `submission.pth` in working directory using `test.py` and submit to Gradescope leaderboard.

```shell
# Submit the generated ./submission.pth to Gradescope
python test.py --submit --ckpt_path CKPT_PATH
```

## Metrics
Table below shows final metrics for SetVAE and our model (**MMD-CD** is scaled by 10<sup>3</sup> and **MMD-EMD**, **COV**, **1-NNA** by 10<sup>2</sup>). SetVAE is trained for 8000 epochs and our model is trained for 2000 epochs.

| Category  | Model | MMD(↓) CD | MMD(↓) EMD | COV(↑) CD | COV(↑) EMD | 1-NNA(↓) CD | 1-NNA(↓) EMD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Airplane | SetVAE | 0.199 | 3.07 | 43.45 | 44.93 | 75.31 | 77.65 |
|  | Ours     | 0.224 | 3.45 | 38.27 | 36.79 | - | - |

