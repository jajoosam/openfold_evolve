import os
import sys
import subprocess
import numpy as np
configs = [
        {
        "lr": 3e-4,
        "iterations": 200,
        "binarized": True,
        "name": "0",
        "dropout": 0.15,
        "noise": 0.01,
    },
    {
        "lr": 3e-3,
        "iterations": 200,
        "binarized": True,
        "name": "1",
        "dropout": 0.15,
        "noise": 0.01,
    },
    {
        "lr": 3e-3,
        "iterations": 200,
        "binarized": True,
        "name": "2",
        "dropout": 0.15,
        "noise": 0.1,
    },
    {
        "lr": 3e-4,
        "iterations": 200,  
        "binarized": True,
        "name": "2",
        "dropout": 0.15,
        "noise": 0.1,
    },
]


for config in configs:
    cmd = [sys.executable, "setup_run.py"]
    for key, value in config.items():
        if(type(value) == bool):
            cmd.extend([f"--{key}"])
        else:
            cmd.extend([f"--{key}", str(value)])
    cmd.extend(["--seed", str(np.random.randint(0, 10000))])
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
