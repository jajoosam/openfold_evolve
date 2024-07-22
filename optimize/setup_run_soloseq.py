import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openfold.config import model_config

from openfold.data.tools import hhsearch, hmmsearch
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.utils.script_utils import prep_output
from openfold.np import protein
from openfold.utils.import_weights import import_jax_weights_, import_openfold_weights_
from openfold.utils.tensor_utils import tensor_tree_map
from scripts.precompute_embeddings import EmbeddingGenerator


from targets_solo import names, tags, seqs

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

config = model_config("seq_model_esm1b_ptm", True, False)

import argparse

parser = argparse.ArgumentParser(description='For sweeps')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--iterations', type=int, default=100)
parser.add_argument('--sam', type=bool, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--friendly_sam', type=bool, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--rho', type=float, default=0.05)
parser.add_argument('--binarized', type=bool, default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--dropout', type=float, default=0.15)
parser.add_argument('--noise', type=float, default=0.1)
parser.add_argument('--name', type=str, default="0")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--init', type=str, default="ones")

parser_args = parser.parse_args()
print(parser_args.binarized)

args = {
    "config_preset": "seq_model_esm1b_ptm",
    "hmmsearch_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/hmmsearch",
    "hhblits_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/hhblits",
    "jackhmmer_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/jackhmmer",
    "hmmbuild_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/hmmbuild",
    "kalign_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/kalign",
    "pdb_seqres_database_path": "../data/pdb_seqres/pdb_seqres.txt",
    "template_mmcif_dir": "../data/pdb_mmcif/mmcif_files/",
    "max_template_date": "3000-01-01",
    "max_hits": 4,
    "mgnify_database_path": "../data/mgnify/mgy_clusters_2022_05.fa",
    "bfd_database_path": None,
    "uniref30_database_path": "../data/uniref30/UniRef30_2021_03",
    "uniref90_database_path": "../data/uniref90/uniref90.fasta",
    "uniprot_database_path": "../data/uniprot/uniprot_sprot.fasta",
    "output_dir": "../output",
    "alignment_dir": "../alignment_dir",
    "cpus": 28,
    "use_precomputed_alignments": True,
    "model_device": "cuda:0",
    "use_single_seq_mode": True,
}

args["openfold_checkpoint_path"] = (
    "../openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt"
)


def get_model_basename(model_path):
    return os.path.splitext(
                os.path.basename(
                    os.path.normpath(model_path)
                )
            )[0]


model_basename = get_model_basename(args["openfold_checkpoint_path"])
model_version = "_".join(model_basename.split("_")[1:])



data_processor = setup_data_processor(None, soloseq=True)
feature_processor = feature_pipeline.FeaturePipeline(config["data"])

name = "7BNY"
target_index = names.index(name)


precompute_alignments(
    tags[target_index], seqs[target_index], args["alignment_dir"], args, soloseq=True
)

feature_dict = generate_feature_dict(
    tags[target_index],
    seqs[target_index],
    args["alignment_dir"],
    data_processor,
    args,
    soloseq=True
)

processed_feature_dict = feature_processor.process_features(
    feature_dict, mode="predict", is_multimer=False
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

seed = parser_args.seed


config_run = {
    "args": args,
    "loss_weights": {
        "weighted_ptm_score": 0,  # 1 is ~ 0.5
        "dropout_rate": 0,  # 1 is ~ 0.33
        "dropout_binarize": 0, # 1 is ~2
        "plddt": 0,
        "pae": 0,  # 1 is ~15
        "distogram": 0,  # 1 is ~1000
        "delta_distogram": 0,  # 1 is ~300
        "history": 0, # 1 is ~10
        "fape_history": 0, # 1 is ~2
        "joint_pae_fape": 0, # 1 is ~sqrt(500)
        "nll": 1,
        "nll_ptm_mul": 0,
    },
    "init": parser_args.init,
    "dropout_rate": parser_args.dropout,
    "learning_rate": parser_args.lr,
    "history_alpha": 3,
    "target": names[target_index],
    "recycles": 0,
    "seed_fixed": True,
    "noise": parser_args.noise,
    "binarized": parser_args.binarized,
    "msa_bias": False,
    "iterations": parser_args.iterations,
    "sam": parser_args.sam,
    "friendly_sam": parser_args.friendly_sam,
    "rho": parser_args.rho,
    "only_first_dropout": False,
    "soloseq": True
}

dropout_config = {
    "NOISE_FACTOR": config_run["noise"],
    "BINARIZE": config_run["binarized"],
    "DROPOUT_RATE": config_run["dropout_rate"],
    "DISABLED": config_run["msa_bias"],
    "ONLY_FIRST_DROPOUT": config_run["only_first_dropout"],
}

with open("tmp_dropout_config.json", "w") as config_file:
    json.dump(dropout_config, config_file)
time.sleep(5)

chains = []

curr_idx = 0
for s in seqs[target_index]:
    start = curr_idx
    curr_idx += len(s)
    end = curr_idx
    chains.append({
        "seq": s,
        "start": start,
        "end": end
    })
from openfold.model.model import AlphaFold

model = AlphaFold(config)
d = torch.load(args["openfold_checkpoint_path"])
if "ema" in d:
    d = d["ema"]["params"]
import_openfold_weights_(model=model, state_dict=d)
model = model.to(args["model_device"])

template_enabled = model.config["template"]["enabled"]

model.config["template"]["enabled"] = template_enabled and any(
    ["template_" in k for k in processed_feature_dict]
)

run_optimization(
    num_residues=num_residues,
    channels_msa=channels_msa,
    channels_msa_extra=channels_msa_extra,
    channels_pair=channels_pair,
    dropout_rate=config_run["dropout_rate"],
    processed_feature_dict=processed_feature_dict,
    iterations=config_run["iterations"],
    name=names[target_index] + "_" + parser_args.name,
    seed=seed,
    model=model,
    feature_dict=feature_dict,
    feature_processor=feature_processor,
    args=args,
    config=config_run,
    chains=chains
)
