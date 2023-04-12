import os
import subprocess


def configure_nccl():
    if os.getenv("LAUNCH_SITE")=='hhe':
        print('HHE init')
        os.environ["NCCL_IB_DISABLE"] = "0"
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_SOCKET_IFNAME"]= "eth0"
        os.environ["NCCL_IB_CUDA_SUPPORT"] = "1"
        os.environ["NCCL_IB_HCA"] =  "mlx5_0,mlx5_2" # "mlx5_0,mlx5_1,mlx5_2,mlx5_5"
        os.environ["NCCL_IB_GID_INDEX"] = "0"    ## change from 3 to 0
        os.environ["NCCL_IB_TC"] = "106"
        os.environ["NCCL_NET_GDR_READ"] = "1"
        os.environ["NCCL_TREE_THRESHOLD"] = "0"
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["CUDA_CACHE_MAXSIZE"] = "2147483647"
        os.environ["CUDA_CACHE_PATH"] = "/data/.cuda_cache"
    elif os.getenv("LAUNCH_SITE")=='hhd':
        print('HHD init')
        # os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
        #os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_IB_HCA"] = subprocess.getoutput(
        "cd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
        "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
        "| grep v >/dev/null && echo $i ; done; > /dev/null"
        )
        os.environ["NCCL_IB_GID_INDEX"] = "3"
        os.environ["NCCL_IB_TC"] = "106"

    elif os.getenv("LAUNCH_SITE")=='hhb':
        print('HHB init')
        os.environ["NCCL_IB_HCA"] = subprocess.getoutput(
        "cd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
        "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
        "| grep v >/dev/null && echo $i ; done; > /dev/null"
        )
        os.environ["NCCL_IB_GID_INDEX"] = "3"
        os.environ["NCCL_IB_TC"] = "106"
        os.environ["NCCL_DEBUG"] = "INFO"
    else:
        return