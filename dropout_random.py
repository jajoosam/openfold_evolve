from openfold.model.model import AlphaFold
from openfold.config import model_config
import torch
import torch.nn as nn
from openfold.data.tools import hhsearch, hmmsearch
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.utils.script_utils import (load_models_from_command_line, parse_fasta, run_model,
                                         prep_output, relax_protein)
from openfold.np import protein
from openfold.utils.import_weights import (
    import_jax_weights_,
    import_openfold_weights_
)

import numpy as np
import wandb

import os
import logging

import subprocess
import re

from openfold.utils.tensor_utils import tensor_tree_map


logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

config = model_config(
    "model_1_multimer_v3",
    True,
    False
)


args = {
    "config_preset": "model_1_multimer_v3",
    "hmmsearch_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/hmmsearch",
    "hhblits_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/hhblits",
    "jackhmmer_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/jackhmmer",
    "hmmbuild_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/hmmbuild",
    "kalign_binary_path": "/home/ubuntu/miniforge3/envs/openfold_env/bin/kalign",
    "pdb_seqres_database_path": "data/pdb_seqres/pdb_seqres.txt",
    "template_mmcif_dir": "data/pdb_mmcif/mmcif_files",
    "max_template_date": "3000-01-01",
    "max_hits": 4,
    "mgnify_database_path": "data/mgnify/mgy_clusters_2022_05.fa",
    "bfd_database_path": None,
    "uniref30_database_path": "data/uniref30/UniRef30_2021_03",
    "uniref90_database_path": "data/uniref90/uniref90.fasta",
    "uniprot_database_path": "data/uniprot/uniprot_sprot.fasta",
    "output_dir": "output",
    "alignment_dir": "alignment_dir",
    "cpus": 28,
    "use_precomputed_alignments": True,
    "model_device": "cuda:0",
}

args["jax_param_path"] = os.path.join(
    "openfold", "resources", "params", "params",
    "params_" + args["config_preset"] + ".npz"
)

def get_model_basename(model_path):
    return os.path.splitext(
                os.path.basename(
                    os.path.normpath(model_path)
                )
            )[0]


model_basename = get_model_basename(args["jax_param_path"])
model_version = "_".join(model_basename.split("_")[1:])
model = AlphaFold(config)
# model = model.eval()
import_jax_weights_(
    model, args["jax_param_path"], version=model_version
)

model = model.to(args["model_device"])
model = model.eval()



template_searcher = hmmsearch.Hmmsearch(
    binary_path=args["hmmsearch_binary_path"],
    hmmbuild_binary_path=args["hmmbuild_binary_path"],
    database_path=args["pdb_seqres_database_path"],
)

template_featurizer = templates.HmmsearchHitFeaturizer(
    mmcif_dir=args["template_mmcif_dir"],
    max_template_date=args["max_template_date"],
    max_hits=config["data"]["predict"]["max_templates"],
    kalign_binary_path=args["kalign_binary_path"],
)


alignment_runner = data_pipeline.AlignmentRunner(
    jackhmmer_binary_path=args["jackhmmer_binary_path"],
    hhblits_binary_path=args["hhblits_binary_path"],
    uniref90_database_path=args["uniref90_database_path"],
    mgnify_database_path=args["mgnify_database_path"],
    bfd_database_path=args["bfd_database_path"],
    uniref30_database_path=args["uniref30_database_path"],
    uniprot_database_path=args["uniprot_database_path"],
    template_searcher=template_searcher,
    no_cpus=args["cpus"],
)


data_processor = data_pipeline.DataPipeline(
    template_featurizer=template_featurizer,
)

data_processor = data_pipeline.DataPipelineMultimer(
    monomer_data_pipeline=data_processor,
)

feature_processor = feature_pipeline.FeaturePipeline(config["data"])

def precompute_alignments(tags, seqs, alignment_dir, args):
    for tag, seq in zip(tags, seqs):
        tmp_fasta_path = os.path.join(args["output_dir"], f"tmp_{os.getpid()}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)

        if args["use_precomputed_alignments"] is None:
            logger.info(f"Generating alignments for {tag}...")

            os.makedirs(local_alignment_dir, exist_ok=True)

            if "multimer" in args["config_preset"]:
                template_searcher = hmmsearch.Hmmsearch(
                    binary_path=args["hmmsearch_binary_path"],
                    hmmbuild_binary_path=args["hmmbuild_binary_path"],
                    database_path=args["pdb_seqres_database_path"],
                )
            else:
                template_searcher = hhsearch.HHSearch(
                    binary_path=args["hhsearch_binary_path"],
                    databases=[args["pdb70_database_path"]],
                )


            alignment_runner = data_pipeline.AlignmentRunner(
                jackhmmer_binary_path=args["jackhmmer_binary_path"],
                hhblits_binary_path=args["hhblits_binary_path"],
                uniref90_database_path=args["uniref90_database_path"],
                mgnify_database_path=args["mgnify_database_path"],
                bfd_database_path=args["bfd_database_path"],
                uniref30_database_path=args["uniref30_database_path"],
                # uniclust30_database_path=args["uniclust30_database_path"],
                uniprot_database_path=args["uniprot_database_path"],
                template_searcher=template_searcher,
                use_small_bfd=args["bfd_database_path"] is None,
                no_cpus=args["cpus"]
            )

            alignment_runner.run(
                tmp_fasta_path, local_alignment_dir
            )
            logger.info("Alignment done!")
        else:
            logger.info(
                f"Using precomputed alignments for {tag} at {alignment_dir}..."
            )

        # Remove temporary FASTA file
        os.remove(tmp_fasta_path)

def generate_feature_dict(
    tags,
    seqs,
    alignment_dir,
    data_processor,
    args,
):
    tmp_fasta_path = os.path.join(args["output_dir"], f"tmp_{os.getpid()}.fasta")

    if "multimer" in args["config_preset"]:
        with open(tmp_fasta_path, "w") as fp:
            fp.write(
                '\n'.join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)])
            )
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path, alignment_dir=alignment_dir,
        )
    elif len(seqs) == 1:
        tag = tags[0]
        seq = seqs[0]
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path,
            alignment_dir=local_alignment_dir,
            seqemb_mode=args["use_single_seq_mode"],
        )
    else:
        with open(tmp_fasta_path, "w") as fp:
            fp.write(
                '\n'.join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)])
            )
        feature_dict = data_processor.process_multiseq_fasta(
            fasta_path=tmp_fasta_path, super_alignment_dir=alignment_dir,
        )

    # Remove temporary FASTA file
    os.remove(tmp_fasta_path)

    return feature_dict

names = [
    "H1106",
    "T1173",
    "T1123",
    "H1144"
]
tags = [
    [
        "H1106-1",
        "H1106-2",
    ],
    [
        "T1173-1",
        "T1173-2",
        "T1173-3",
    ],
    [
        "T1123-1",
        "T1123-2",
    ],
    [
        "H1144-1",
        "H1144-2",
    ]
]
seqs = [
    [
        "MSRIITAPHIGIEKLSAISLEELSCGLPDRYALPPDGHPVEPHLERLYPTAQSKRSLWDFASPGYTFHGLHRAQDYRRELDTLQSLLTTSQSSELQAAAALLKCQQDDDRLLQIILNLLHKV",
        "MNITLTKRQQEFLLLNGWLQLQCGHAERACILLDALLTLNPEHLAGRRCRLVALLNNNQGERAEKEAQWLISHDPLQAGNWLCLSRAQQLNGDLDKARHAYQHYLELKDHNESP"
    ],
    [
        "NASINFVATEAHTASAGGAKIIFNTTNNGATGSTEKVVIDQNGNVGVGVGAPTAKMDVNGGIKQPNYGIISAVRNSGGVTASMPWTNAYVLAHQGEMHQWVAGGPILQDSVTGCNAGPDAGVKFDSIATSWGGPYKVIFHTTGSNGAIHLEWSGWQVSLKNSAGTELAIGMGQVFATLHYDPAVSNWRVEHMFGRINNTNFTCW",
        "NASINFVATEAHTASAGGAKIIFNTTNNGATGSTEKVVIDQNGNVGVGVGAPTAKMDVNGGIKQPNYGIISAVRNSGGVTASMPWTNAYVLAHQGEMHQWVAGGPILQDSVTGCNAGPDAGVKFDSIATSWGGPYKVIFHTTGSNGAIHLEWSGWQVSLKNSAGTELAIGMGQVFATLHYDPAVSNWRVEHMFGRINNTNFTCW",
        "NASINFVATEAHTASAGGAKIIFNTTNNGATGSTEKVVIDQNGNVGVGVGAPTAKMDVNGGIKQPNYGIISAVRNSGGVTASMPWTNAYVLAHQGEMHQWVAGGPILQDSVTGCNAGPDAGVKFDSIATSWGGPYKVIFHTTGSNGAIHLEWSGWQVSLKNSAGTELAIGMGQVFATLHYDPAVSNWRVEHMFGRINNTNFTCW"
    ],
    [
        "MHHHHHHHHHHSETTYTGPRSIVTPETPIGPSSYPMTPSSLVLMAGYFSGPEISDNFGKYMPLLFQQNTSKVTFRSGSHTIKIVSMVLVDRLMWLDKHFNQYTNEPDGVFGDVGNVFVDNDNVAKVITMSGSSAPANRGATLMLCRATKNIQTFNFAATVYIPAYKVKDGAGGKDVVLNVAQWEANKTLTYPAIPKDTYFMVVTMGGASFTIQRYVVYNEGIGDGLELPAFWGKYLSQLYGFSWSSPTYACVTWEPIYAEEGIPHR",
        "MHHHHHHHHHHSETTYTGPRSIVTPETPIGPSSYPMTPSSLVLMAGYFSGPEISDNFGKYMPLLFQQNTSKVTFRSGSHTIKIVSMVLVDRLMWLDKHFNQYTNEPDGVFGDVGNVFVDNDNVAKVITMSGSSAPANRGATLMLCRATKNIQTFNFAATVYIPAYKVKDGAGGKDVVLNVAQWEANKTLTYPAIPKDTYFMVVTMGGASFTIQRYVVYNEGIGDGLELPAFWGKYLSQLYGFSWSSPTYACVTWEPIYAEEGIPHR"
    ],
    [
        "GLEKDFLPLYFGWFLTKKSSETLRKAGQVFLEELGNHKAFKKELRHFISGDEPKEKLELVSYFGKRPPGVLHCTTKFCDYKAAGAEEYAQQEVVKRSYGKAFKLSISALFVTPKTAGAQVVLTDQELQLWPSDLDKPSASEGLPPGSRAHVTLGCAADVQPVQTGLDLLDILQQVKGGSQGEAVGELPRGKLYSLGKGRWMLSLTKKMEVKAIFTGYYG",
        "EVQLEESGGGLVQPGGSLRLSCAASGFTFSSYVMSWVRQAPGKGLEWVSDINSGGSRTYYTDSVKGRFTISRDNAKNTLYLQMNSLKPEDTAVYYCARDSLLSTRYLHTSERGQGTQVTVSS"
    ]
]

target_index =3

precompute_alignments(tags[target_index], seqs[target_index], args["alignment_dir"], args)

# TODO: generate_feature_dict 

feature_dict = generate_feature_dict(
    tags[target_index],
    seqs[target_index],
    args["alignment_dir"],
    data_processor,
    args,
)

processed_feature_dict = feature_processor.process_features(
    feature_dict, mode='predict', is_multimer=True
)

processed_feature_dict = {
    k: torch.as_tensor(v).to(args["model_device"])
    for k, v in processed_feature_dict.items()
}

template_enabled = model.config["template"]["enabled"]

model.config["template"]["enabled"] = template_enabled and any([
    "template_" in k for k in processed_feature_dict
])

num_residues = 341
channels_msa = 256
channels_msa_extra = 64
channels_pair = 128
dropout_rate = 0.15


class DropoutMaskOptimizer(nn.Module):
    def __init__(self, num_residues, channels_msa, channels_msa_extra, channels_pair):
        super(DropoutMaskOptimizer, self).__init__()
        # sigmoid shenanigans
        self.msa_dropout_mask = nn.Parameter((1 - dropout_rate)*2 * (torch.rand(num_residues, channels_msa) * 8) - 4)
        self.pair_dropout_mask = nn.Parameter((1 - dropout_rate)*2 * (torch.rand(4, num_residues, channels_pair) * 8) - 4)
        self.extra_msa_dropout_mask = nn.Parameter((1 - dropout_rate)*2 * (torch.rand(num_residues, channels_msa_extra) * 8) - 4)
        

    def forward(self):
        return {
            "msa_dropout_mask": self.msa_dropout_mask,
            "pair_dropout_mask": self.pair_dropout_mask,
            "extra_msa_dropout_mask": self.extra_msa_dropout_mask,
        }



def run_mmalign_and_get_tmscore(structure_1, structure_2):
    # Define the command
    command = ["./../USalign/MMalign", structure_1, structure_2]
    
    # Run the command and capture the output
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Check if the command was successful
    if result.returncode != 0:
        print("Error running MM-align:")
        print(result.stderr)
        return None
    
    # Extract the TM-score from the output
    output = result.stdout
    print(output)
    tm_score_match = re.search(r"TM-score=\s+([\d\.]+)", output)
    
    if tm_score_match:
        tm_score = float(tm_score_match.group(1))
        return torch.tensor(tm_score)
    else:
        return torch.nan

# Example usage


# # Initialize the DropoutMaskOptimizer
dropout_mask_optimizer = DropoutMaskOptimizer(num_residues, channels_msa, channels_msa_extra, channels_pair).to(args["model_device"]).train()


run = wandb.init(
    project="af_dropout",
    config={
        "args"    : args,
        "loss_weights": {
            "weighted_ptm_score": 0,
            "dropout_rate": 1e+2,
            "dropout_binarize": 1e-5,
            "plddt": 1e+0,
        },
        "learning_rate": 3e-2,
        "target": names[target_index],
        "recycles": 0
    }
)



# # Pass the learnable parameters to the optimizer
optimizer = torch.optim.Adam(dropout_mask_optimizer.parameters(), lr=run.config["learning_rate"])

# msa_dropout_mask = torch.ones(num_residues, channels_msa)
# pair_dropout_mask = torch.ones(4, num_residues, channels_pair)
# extra_msa_dropout_mask = torch.ones(num_residues, channels_msa_extra)

def generate_plausible_dropout_masks(num_residues, channels_msa, channels_msa_extra, channels_pair, seed):
    torch.manual_seed(seed)
    msa_dropout_mask = (torch.rand(num_residues, channels_msa) > (1 - dropout_rate)).float().to(args["model_device"])
    pair_dropout_mask = (torch.rand(4, num_residues, channels_pair) > (1 - dropout_rate)).float().to(args["model_device"])
    extra_msa_dropout_mask = (torch.rand(num_residues, channels_msa_extra) > (1 - dropout_rate)).float().to(args["model_device"])
    return {
        "msa_dropout_mask": msa_dropout_mask,
        "pair_dropout_mask": pair_dropout_mask,
        "extra_msa_dropout_mask": extra_msa_dropout_mask,
    }

# Training loop
for i in range(10000):
    
    processed_feature_dict.update(generate_plausible_dropout_masks(num_residues, channels_msa, channels_msa_extra, channels_pair, seed=i))
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(processed_feature_dict)

    weighted_ptm_score =  out["weighted_ptm_score"]
    plddt = out["plddt"].mean()
    print(f"Weighted ptm score: {weighted_ptm_score}")

    # sigmoided_msa_dropout_mask = torch.sigmoid(processed_feature_dict["msa_dropout_mask"])
    # sigmoided_pair_dropout_mask = torch.sigmoid(processed_feature_dict["pair_dropout_mask"])
    # sigmoided_extra_msa_dropout_mask = torch.sigmoid(processed_feature_dict["extra_msa_dropout_mask"])

    # msa_dropout_square_sum = ((sigmoided_msa_dropout_mask - 2)**2).sum()
    # pair_dropout_square_sum = ((sigmoided_pair_dropout_mask - 2)**2).sum()
    # extra_msa_dropout_square_sum = ((sigmoided_extra_msa_dropout_mask - 2)**2).sum()

    # msa_dropout_rate = 1 -sigmoided_msa_dropout_mask.mean()
    # pair_dropout_rate = 1 - sigmoided_pair_dropout_mask.mean()
    # extra_msa_dropout_rate = 1 -sigmoided_extra_msa_dropout_mask.mean()


    # ptm_loss = -1 * run.config["loss_weights"]["weighted_ptm_score"] * weighted_ptm_score 
    # dropout_rate_loss = run.config["loss_weights"]["dropout_rate"] * ((msa_dropout_rate - dropout_rate)**2 + (pair_dropout_rate - dropout_rate)**2 + (extra_msa_dropout_rate - dropout_rate)**2).sum()
    # dropout_binarize_loss = run.config["loss_weights"]["dropout_binarize"] * (msa_dropout_square_sum + pair_dropout_square_sum + extra_msa_dropout_square_sum)
    # plddt_loss = -1 * plddt

    # loss = ptm_loss + dropout_rate_loss + dropout_binarize_loss + plddt_loss

    # print(f"Loss: {loss}")
    # print(f"out: {out.keys()}")
    # print(f"pae: {out['final_atom_positions']}")
    # loss.backward()

    

    # optimizer.step()
    # optimizer.zero_grad()

    processed_feature_dict_detached = tensor_tree_map(
    lambda x: np.array(x[..., -1].detach().cpu()),
    processed_feature_dict
    )

    out_detached = tensor_tree_map(lambda x: np.array(x.detach().cpu()), out)

    unrelaxed_protein = prep_output(
        out_detached,
        processed_feature_dict_detached,
        feature_dict,
        feature_processor,
        args["config_preset"],
        200,
        False
    )


    unrelaxed_output_path = os.path.join(
        args["output_dir"], f'{names[target_index]}_unrelaxed.pdb'
    )

    with open(unrelaxed_output_path, 'w') as fp:
            fp.write(protein.to_pdb(unrelaxed_protein))

    # NOTE: order of args is important here
    tm_score = run_mmalign_and_get_tmscore(f"references/{names[target_index]}.pdb", unrelaxed_output_path)
    predicted_tm_score = out["weighted_ptm_score"]
    print(f"plddt: {out['plddt'].mean()}")
    wandb.log({"plddt": out["plddt"].mean(), "tm_score": tm_score, "predicted_tm_score": predicted_tm_score})

    # wandb.log({"plddt": out["plddt"].mean(), "tm_score": tm_score, "predicted_tm_score": predicted_tm_score, "ptm_loss": ptm_loss, "dropout_rate_loss": dropout_rate_loss, "dropout_binarize_loss": dropout_binarize_loss, "loss": loss})

