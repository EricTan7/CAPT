### Train
Refer to the command in `train_baseline_fs.sh`

`train.py` is the plain training script. 
`train_wandb.py` takes W&B to log the training.
`train_sweep_hyper.py` takes W&B to automatically sweep hyperparameters, if used, you need to modify the params to be swept in the file.