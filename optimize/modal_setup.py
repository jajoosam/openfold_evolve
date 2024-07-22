# c_m = 256
# c_z = 128

run_id = "dropout_0.15_100_evolve_intersection_individual"

import modal
from modal import Image, Volume
import torch
import math

#TODO: be less stupid about order of steps in the image building instructions
openfold_image = (Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel", add_python="3.9")
        .run_commands("mkdir openfold_evolve")
        .copy_local_dir("../../openfold", "openfold_evolve/openfold")
        .copy_local_dir("../../USalign", "openfold_evolve/USalign")
)

app = modal.App("openfold_evolve")

@app.function(image=openfold_image)
def my_function():
    print("Hi")