import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold.model.model import AlphaFold
from openfold.config import model_config

from openfold.data.tools import hhsearch, hmmsearch
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.utils.script_utils import prep_output
from openfold.np import protein
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.tensor_utils import tensor_tree_map

from targets import names, tags, seqs

import numpy as np
import wandb

import os
import logging
import json

from utils import (
    setup_data_processor,
    precompute_alignments,
    generate_feature_dict,
)

from run_optimization import run_optimization

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

config = model_config("model_1_multimer_v3", True, False)


args = {
    "config_preset": "model_1_multimer_v2",
    "hmmsearch_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/hmmsearch",
    "hhblits_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/hhblits",
    "jackhmmer_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/jackhmmer",
    "hmmbuild_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/hmmbuild",
    "kalign_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/kalign",
    "pdb_seqres_database_path": "data/pdb_seqres/pdb_seqres.txt",
    "template_mmcif_dir": "data/pdb_mmcif/mmcif_files/",
    "max_template_date": "3000-01-01",
    "max_hits": 4,
    "mgnify_database_path": "data/mgnify/mgy_clusters_2022_05.fa",
    "bfd_database_path": None,
    "uniref30_database_path": "data/uniref30/UniRef30_2021_03",
    "uniref90_database_path": "data/uniref90/uniref90.fasta",
    "uniprot_database_path": "data/uniprot/uniprot_sprot.fasta",
    "output_dir": "../output",
    "alignment_dir": "../alignment_dir",
    "cpus": 28,
    "use_precomputed_alignments": True,
    "model_device": "cuda:0",
}

args["jax_param_path"] = (
    f"../openfold/resources/params/v2.2/params_{args['config_preset']}.npz"
)


def get_model_basename(model_path):
    return os.path.splitext(
                os.path.basename(
                    os.path.normpath(model_path)
                )
            )[0]


model_basename = get_model_basename(args["jax_param_path"])
model_version = "_".join(model_basename.split("_")[1:])



data_processor = setup_data_processor(None)
feature_processor = feature_pipeline.FeaturePipeline(config["data"])

name = "H1144"
target_index = names.index(name)

print(target_index)


precompute_alignments(
    tags[target_index], seqs[target_index], args["alignment_dir"], args
)

feature_dict = generate_feature_dict(
    tags[target_index],
    seqs[target_index],
    args["alignment_dir"],
    data_processor,
    args,
)

processed_feature_dict = feature_processor.process_features(
    feature_dict, mode="predict", is_multimer=True
)

processed_feature_dict = {
    k: torch.as_tensor(v).to(args["model_device"])
    for k, v in processed_feature_dict.items()
}


num_residues = np.array([len(seq) for seq in seqs[target_index]]).sum()
channels_msa = 256
channels_msa_extra = 64
channels_pair = 128
dropout_rate = 0.15

seed = 0

iterations = 100

config_run = {
    "args": args,
    "loss_weights": {
        "weighted_ptm_score": 1,  # 1 is ~ 0.5
        "dropout_rate": 0,  # 1 is ~ 0.33
        "dropout_binarize": 0, # 1 is ~2
        "plddt": 0,
        "pae": 0,  # 1 is ~15
        "distogram": 0,  # 1 is ~1000
        "delta_distogram": 0,  # 1 is ~300
        "history": 0, # 1 is ~10
        "fape_history": 0, # 1 is ~2
        "joint_pae_fape": 0, # 1 is ~sqrt(500)
    },
    "learning_rate": 1e-2,
    "history_alpha": 3,
    "target": names[target_index],
    "recycles": 0,
    "seed_fixed": True,
    "noise": 0,
    "binarized": True,
    "msa_bias": False
}

dropout_config = {
    "NOISE_FACTOR": config_run["noise"],
    "BINARIZE": config_run["binarized"],
    "DROPOUT_RATE": dropout_rate,
}

with open("tmp_dropout_config.json", "w") as config_file:
    json.dump(dropout_config, config_file)


model = AlphaFold(config)
import_jax_weights_(model, args["jax_param_path"], version=model_version)
model = model.to(args["model_device"])
model = model.train()

template_enabled = model.config["template"]["enabled"]

model.config["template"]["enabled"] = template_enabled and any(
    ["template_" in k for k in processed_feature_dict]
)

run_optimization(
    num_residues=num_residues,
    channels_msa=channels_msa,
    channels_msa_extra=channels_msa_extra,
    channels_pair=channels_pair,
    dropout_rate=dropout_rate,
    processed_feature_dict=processed_feature_dict,
    iterations=iterations,
    name=names[target_index],
    seed=seed,
    model=model,
    feature_dict=feature_dict,
    feature_processor=feature_processor,
    args=args,
    config=config_run,
)
