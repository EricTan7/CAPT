import os
import json
import argparse
import numpy as np
import wandb


def average(root):
    # Suppose the save path is:
    # root / [dataset] / [model] / [backbone + shots] / [hyper(lr+iter+rank+alpha)] / [seed]
    for shots in os.listdir(root):
        shots_dir = os.path.join(root, shots)
        for hyper in os.listdir(shots_dir):
            hyper_dir = os.path.join(shots_dir, hyper)
            test_acc, test_acc_wiseft = [], []
            for seed in os.listdir(hyper_dir):
                seed_dir = os.path.join(hyper_dir, seed)
                test_path = os.path.join(seed_dir, "test.json")
                if os.path.exists(test_path):
                    with open(test_path, 'r') as f:
                        res = json.load(f)
                        test_acc.append(res["test acc"])
                        test_acc_wiseft.append(res["test acc (wiseft_0.5)"])
            # mean, std
            test_acc_mean = np.mean(test_acc)
            test_acc_wiseft_mean = np.mean(test_acc_wiseft)
            test_acc_std = np.std(test_acc)
            test_acc_wiseft_std = np.std(test_acc_wiseft)

            save_dict = {
                'test_acc': test_acc,
                'test_acc_wiseft': test_acc_wiseft,
                'test_acc_mean': test_acc_mean,
                'test_acc_wiseft_mean': test_acc_wiseft_mean,
                'test_acc_std': test_acc_std,
                'test_acc_wiseft_std': test_acc_wiseft_std
            }

            wandb.log({'test_acc_mean': test_acc_mean,
                'test_acc_wiseft_mean': test_acc_wiseft_mean,
                'test_acc_std': test_acc_std,
                'test_acc_wiseft_std': test_acc_wiseft_std})

            print(save_dict)
            save_path = os.path.join(hyper_dir, "average.json")
            with open(save_path, "w") as f:
                json.dump(save_dict, f)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--root", type=str, default="/mnt/sdb/tanhao/recognition/", help="path to dataset")
#     args = parser.parse_args()
#
#     average(args.root)