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
    "output_dir": "output",
    "alignment_dir": "alignment_dir",
    "cpus": 28,
    "use_precomputed_alignments": True,
    "model_device": "cuda:0",
}

args["jax_param_path"] = os.path.join(
    "openfold", "resources", "params", "v2.2",
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
model = model.train()



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
    template_featurizer=None,
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
    "H1144",
    "H1106",
    "T1173",
]
tags = [
    [
        "H1144-1",
        "H1144-2",
    ],
    [
        "H1106-1",
        "H1106-2",
    ],
    [
        "T1173-1",
        "T1173-2",
        "T1173-3",
    ],
]
seqs = [
    [
        "GLEKDFLPLYFGWFLTKKSSETLRKAGQVFLEELGNHKAFKKELRHFISGDEPKEKLELVSYFGKRPPGVLHCTTKFCDYKAAGAEEYAQQEVVKRSYGKAFKLSISALFVTPKTAGAQVVLTDQELQLWPSDLDKPSASEGLPPGSRAHVTLGCAADVQPVQTGLDLLDILQQVKGGSQGEAVGELPRGKLYSLGKGRWMLSLTKKMEVKAIFTGYYG",
        "EVQLEESGGGLVQPGGSLRLSCAASGFTFSSYVMSWVRQAPGKGLEWVSDINSGGSRTYYTDSVKGRFTISRDNAKNTLYLQMNSLKPEDTAVYYCARDSLLSTRYLHTSERGQGTQVTVSS"
    ],
    [
        "MSRIITAPHIGIEKLSAISLEELSCGLPDRYALPPDGHPVEPHLERLYPTAQSKRSLWDFASPGYTFHGLHRAQDYRRELDTLQSLLTTSQSSELQAAAALLKCQQDDDRLLQIILNLLHKV",
        "MNITLTKRQQEFLLLNGWLQLQCGHAERACILLDALLTLNPEHLAGRRCRLVALLNNNQGERAEKEAQWLISHDPLQAGNWLCLSRAQQLNGDLDKARHAYQHYLELKDHNESP"
    ],
    [
        "NASINFVATEAHTASAGGAKIIFNTTNNGATGSTEKVVIDQNGNVGVGVGAPTAKMDVNGGIKQPNYGIISAVRNSGGVTASMPWTNAYVLAHQGEMHQWVAGGPILQDSVTGCNAGPDAGVKFDSIATSWGGPYKVIFHTTGSNGAIHLEWSGWQVSLKNSAGTELAIGMGQVFATLHYDPAVSNWRVEHMFGRINNTNFTCW",
        "NASINFVATEAHTASAGGAKIIFNTTNNGATGSTEKVVIDQNGNVGVGVGAPTAKMDVNGGIKQPNYGIISAVRNSGGVTASMPWTNAYVLAHQGEMHQWVAGGPILQDSVTGCNAGPDAGVKFDSIATSWGGPYKVIFHTTGSNGAIHLEWSGWQVSLKNSAGTELAIGMGQVFATLHYDPAVSNWRVEHMFGRINNTNFTCW",
        "NASINFVATEAHTASAGGAKIIFNTTNNGATGSTEKVVIDQNGNVGVGVGAPTAKMDVNGGIKQPNYGIISAVRNSGGVTASMPWTNAYVLAHQGEMHQWVAGGPILQDSVTGCNAGPDAGVKFDSIATSWGGPYKVIFHTTGSNGAIHLEWSGWQVSLKNSAGTELAIGMGQVFATLHYDPAVSNWRVEHMFGRINNTNFTCW"
    ],
]

seeds = torch.randint(0, 1000, (3, 10, 3)) # 3 targets, 10 iterations, 3 diff seeds

for target_index in range(1):
    for iteration in range(1):
        precompute_alignments(tags[target_index], seqs[target_index], args["alignment_dir"], args)


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

        num_residues = np.array([len(seq) for seq in seqs[target_index]]).sum()
        channels_msa = 256
        channels_msa_extra = 64
        channels_pair = 128
        dropout_rate = 0.15

        torch.set_grad_enabled(True)
        torch.manual_seed(seeds[target_index][iteration][0])
        class DropoutMaskOptimizer(nn.Module):
            def __init__(self, num_residues, channels_msa, channels_msa_extra, channels_pair):
                super(DropoutMaskOptimizer, self).__init__()
                # sigmoid shenanigans
                # self.msa_dropout_mask = nn.Parameter((1 - dropout_rate)*2 * (torch.rand(num_residues, channels_msa) * 8) - 4)
                # self.pair_dropout_mask = nn.Parameter((1 - dropout_rate)*2 * (torch.rand(4, num_residues, channels_pair) * 8) - 4)
                # self.extra_msa_dropout_mask = nn.Parameter((1 - dropout_rate)*2 * (torch.rand(num_residues, channels_msa_extra) * 8) - 4)

                self.msa_dropout_mask = nn.Parameter((torch.rand(num_residues, channels_msa) - 0.5))
                self.pair_dropout_mask = nn.Parameter((torch.rand(4, num_residues, channels_pair) - 0.5))
                self.extra_msa_dropout_mask = nn.Parameter((torch.rand(num_residues, channels_msa_extra) - 0.5))
                

            def forward(self):
                return {
                    "msa_dropout_mask": self.msa_dropout_mask,
                    "pair_dropout_mask": self.pair_dropout_mask,
                    "extra_msa_dropout_mask": self.extra_msa_dropout_mask,
                }
        
        class CoefficeientOptimizer(nn.Module):
            def __init__(self):
                super(CoefficeientOptimizer, self).__init__()

                self.pae_coeff = nn.Parameter(torch.rand(1)[0])
                self.delta_distogram_coeff = nn.Parameter(torch.rand(1)[0])
                self.dropout_rate_coeff = nn.Parameter(torch.rand(1)[0])
                self.dropout_binarize_coeff = nn.Parameter(torch.rand(1)[0])
                self.plddt_coeff = nn.Parameter(torch.rand(1)[0])
                self.weighted_ptm_coeff = nn.Parameter(torch.rand(1)[0])
                self.joint_change_confidence_coeff = nn.Parameter(torch.rand(1)[0])

            def forward(self):
                return {
                    "pae": self.pae_coeff,
                    "delta_distogram": self.delta_distogram_coeff,
                    "dropout_rate": self.dropout_rate_coeff,
                    "dropout_binarize": self.dropout_binarize_coeff,
                    "plddt": self.plddt_coeff,
                    "weighted_ptm_score": self.weighted_ptm_coeff,
                    "joint_change_confidence": self.joint_change_confidence_coeff,
                }


        def run_mmalign_and_get_tmscore(reference, model):
            # Define the command
            command = ["./../USalign/MMalign", reference, model]
            
            # Run the command and capture the output
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Check if the command was successful
            if result.returncode != 0:
                print("Error running MM-align:")
                print(result.stderr)
                return None
            
            # Extract the TM-score from the output
            output = result.stdout
            tm_score_match = re.search(r"TM-score=\s+([\d\.]+)", output)
            
            if tm_score_match:
                tm_score = float(tm_score_match.group(1))
                return torch.tensor(tm_score)
            else:
                return torch.nan

        # Example usage


        # # Initialize the DropoutMaskOptimizer
        dropout_mask_optimizer = DropoutMaskOptimizer(num_residues, channels_msa, channels_msa_extra, channels_pair).to(args["model_device"]).train()

        coefficient_optimizer = CoefficeientOptimizer().to(args["model_device"]).train()

        run = wandb.init(
            project="af_dropout",
            config={
                "args"    : args,
                "loss_weights": {
                    "weighted_ptm_score": 0, # 1 is ~ 0.5
                    "dropout_rate": 0, # 1 is ~ 0.33
                    "dropout_binarize": 0,
                    "plddt": 0,
                    "pae": 0, # 1 is ~15
                    "distogram": 1, # 1 is ~1000
                    "delta_distogram": 0, # 1 is ~300
                    "joint_change_confidence": 0, # 1 is ~ 2.5
                },
                "learning_rate": 3e-3,
                "target": names[target_index] + "_" + str(iteration),
                "recycles": 0,
                "seed_fixed": True,
                "noise_added": True,
                "binarized": False
            }
        )

        old_masks = {}

        # # Pass the learnable parameters to the optimizer
        optimizer = torch.optim.Adam(dropout_mask_optimizer.parameters(), lr=run.config["learning_rate"])
        optimizer_coeff = torch.optim.Adam(coefficient_optimizer.parameters(), lr=run.config["learning_rate"])
        # msa_dropout_mask = torch.ones(num_residues, channels_msa)
        # pair_dropout_mask = torch.ones(4, num_residues, channels_pair)
        # extra_msa_dropout_mask = torch.ones(num_residues, channels_msa_extra)

        # def compute_drmsd(structure_1, structure_2, mask=None):
        #     def prep_d(structure):
        #         d = structure[..., :, None, :] - structure[..., None, :, :]
        #         d = d ** 2
        #         d = torch.sqrt(torch.sum(d, dim=-1))
        #         return d

        #     d1 = prep_d(structure_1)
        #     d2 = prep_d(structure_2)

        #     drmsd = d1 - d2
        #     drmsd = drmsd ** 2
        #     if(mask is not None):
        #         drmsd = drmsd * (mask[..., None] * mask[..., None, :])
        #     drmsd = torch.sum(drmsd, dim=(-1, -2))
        #     n = d1.shape[-1] if mask is None else torch.min(torch.sum(mask, dim=-1))
        #     drmsd = drmsd * (1 / (n * (n - 1))) if n > 1 else (drmsd * 0.)
        #     drmsd = torch.sqrt(drmsd + 1e-8)

        #     return drmsd

        # prev_atom_positions = None

        reference_pdb = open(f"references/{names[target_index]}.pdb").read()
        reference_structure = protein.from_pdb_string(reference_pdb)

        reference_distogram = torch.zeros(num_residues, num_residues)
        for i in range(num_residues):
            for j in range(num_residues):
                reference_distogram[i, j] = torch.norm(torch.tensor(reference_structure.atom_positions[i][1]) - torch.tensor(reference_structure.atom_positions[j][1]))

        reference_distogram = reference_distogram.to(args["model_device"])
        prev_distogram = None
        prev_pae = None

        # Training loop
        for i in range(150):
            
            processed_feature_dict.update(dropout_mask_optimizer())
            
            if(i > 0):
                delta_msa_dropout_mask = torch.abs(processed_feature_dict["msa_dropout_mask"] - old_masks["msa_dropout_mask"])
                delta_pair_dropout_mask = torch.abs(processed_feature_dict["pair_dropout_mask"] - old_masks["pair_dropout_mask"])
                delta_extra_msa_dropout_mask = torch.abs(processed_feature_dict["extra_msa_dropout_mask"] - old_masks["extra_msa_dropout_mask"])
                print("MSA dropout mask change:", delta_msa_dropout_mask.mean(), torch.abs(processed_feature_dict["msa_dropout_mask"]).mean())
                print("Pair dropout mask change:", delta_pair_dropout_mask.mean(), torch.abs(processed_feature_dict["pair_dropout_mask"]).mean())
                print("Extra MSA dropout mask change:", delta_extra_msa_dropout_mask.mean(), torch.abs(processed_feature_dict["extra_msa_dropout_mask"]).mean())

            old_masks = {
                "msa_dropout_mask": processed_feature_dict["msa_dropout_mask"].clone(),
                "pair_dropout_mask": processed_feature_dict["pair_dropout_mask"].clone(),
                "extra_msa_dropout_mask": processed_feature_dict["extra_msa_dropout_mask"].clone(),
            }
            torch.manual_seed(seeds[target_index][iteration][1])
            np.random.seed(seeds[target_index][iteration][2])

            out = model(processed_feature_dict)
            print(f"Forward pass #{i} complete")
            
            weighted_ptm_score =  out["weighted_ptm_score"]
            plddt = out["plddt"].mean()
            pae = out["predicted_aligned_error"].mean()

            noise_msa = processed_feature_dict["msa_dropout_mask"] + (torch.rand_like(processed_feature_dict["msa_dropout_mask"])-0.5)*3
            noise_pair = processed_feature_dict["pair_dropout_mask"] + (torch.rand_like(processed_feature_dict["pair_dropout_mask"])-0.5)*3
            noise_extra = processed_feature_dict["extra_msa_dropout_mask"] + (torch.rand_like(processed_feature_dict["extra_msa_dropout_mask"])-0.5)*3

            sigmoided_msa_dropout_mask = torch.sigmoid(noise_msa)
            sigmoided_pair_dropout_mask = torch.sigmoid(noise_pair)
            sigmoided_extra_msa_dropout_mask = torch.sigmoid(noise_extra)

            # hard_msa_dropout_mask = (sigmoided_msa_dropout_mask > 0.5).float()
            # hard_pair_dropout_mask = (sigmoided_pair_dropout_mask > 0.5).float()
            # hard_extra_msa_dropout_mask = (sigmoided_extra_msa_dropout_mask > 0.5).float()

            # sigmoided_msa_dropout_mask = sigmoided_msa_dropout_mask + (hard_msa_dropout_mask - sigmoided_msa_dropout_mask).detach()
            # sigmoided_pair_dropout_mask = sigmoided_pair_dropout_mask + (hard_pair_dropout_mask - sigmoided_pair_dropout_mask).detach()
            # sigmoided_extra_msa_dropout_mask = sigmoided_extra_msa_dropout_mask + (hard_extra_msa_dropout_mask - sigmoided_extra_msa_dropout_mask).detach()

            def get_binarization_loss(sigmoided_values):
                loss = -1 * ((sigmoided_values * torch.log(sigmoided_values)) + (1 - sigmoided_values) * torch.log(1 - sigmoided_values)).mean()
                return loss

            msa_dropout_rate = 1 -sigmoided_msa_dropout_mask.mean()
            pair_dropout_rate = 1 - sigmoided_pair_dropout_mask.mean()
            extra_msa_dropout_rate = 1 -sigmoided_extra_msa_dropout_mask.mean()


            # ptm_loss = -1 * run.config["loss_weights"]["weighted_ptm_score"] * weighted_ptm_score 
            # dropout_rate_loss = run.config["loss_weights"]["dropout_rate"] * ((msa_dropout_rate - dropout_rate)**2 + (pair_dropout_rate - dropout_rate)**2 + (extra_msa_dropout_rate - dropout_rate)**2).sum()
            # dropout_binarize_loss = run.config["loss_weights"]["dropout_binarize"] * (get_binarization_loss(sigmoided_msa_dropout_mask) + get_binarization_loss(sigmoided_pair_dropout_mask) + get_binarization_loss(sigmoided_extra_msa_dropout_mask))
            # plddt_loss = -1 * run.config["loss_weights"]["plddt"] * plddt
            # pae_loss = run.config["loss_weights"]["pae"] * pae

            loss_cofficients = coefficient_optimizer()

            ptm_loss = -1 * loss_cofficients["weighted_ptm_score"] * weighted_ptm_score
            dropout_rate_loss = loss_cofficients["dropout_rate"] * ((msa_dropout_rate - dropout_rate)**2 + (pair_dropout_rate - dropout_rate)**2 + (extra_msa_dropout_rate - dropout_rate)**2).sum()
            dropout_binarize_loss = loss_cofficients["dropout_binarize"] * (get_binarization_loss(sigmoided_msa_dropout_mask) + get_binarization_loss(sigmoided_pair_dropout_mask) + get_binarization_loss(sigmoided_extra_msa_dropout_mask))
            plddt_loss = -1 * loss_cofficients["plddt"] * plddt
            pae_loss = loss_cofficients["pae"] * pae


            # if(i > 0):
            #     drmsd_loss = compute_drmsd(
            #         out["final_atom_positions"],
            #         prev_atom_positions,
            #         mask=out["final_atom_mask"],
            #     ).mean()
            #     drmsd_loss = -1 * run.config["loss_weights"]["drmsd"] * drmsd_loss
            # else:
            #     drmsd_loss = 0
            

            # distogram = torch.zeros((num_residues, num_residues))
            # for i in range(num_residues):
            #     for j in range(num_residues):
            #         distogram[i, j] = torch.norm(out["final_atom_positions"][i][1] - out["final_atom_positions"][j][1])
            # distogram = distogram.to(args["model_device"])
            
            atom_positions = out["final_atom_positions"][:, 1, :]  # Shape: (num_residues, 3)

            # Calculate pairwise distances using broadcasting
            distogram = torch.norm(atom_positions[:, None, :] - atom_positions[None, :, :], dim=2)

            # Move the distogram to the desired device
            distogram = distogram.to(args["model_device"])

            print(f"{distogram.shape}")

            
            distogram_loss = run.config["loss_weights"]["distogram"] * torch.norm(distogram - reference_distogram)


            delta_dis_loss = 0
            delta_dis = 0
            if(i > 0):
                delta_dis = torch.norm(distogram - prev_distogram)
                # delta_dis_loss = -1 * run.config["loss_weights"]["delta_distogram"] * delta_dis
                delta_dis_loss = -1 * loss_cofficients["delta_distogram"] * delta_dis
            prev_distogram = distogram.detach().clone()
        
            delta_pae = 0
            if(i > 0):
                delta_pae = pae - prev_pae # more negative = more confidence
            prev_pae = pae.detach().clone()

        
            
            # joint_change_confidence_loss =  run.config["loss_weights"]["joint_change_confidence"] * (delta_pae - delta_dis/50)**2
            joint_change_confidence_loss =  loss_cofficients["joint_change_confidence"] * (delta_pae - delta_dis/50)**2

            distogram_loss.backward(retain_graph=True)
            optimizer_coeff.step()
            optimizer_coeff.zero_grad()
            print(coefficient_optimizer())

            loss =  ptm_loss + dropout_rate_loss + dropout_binarize_loss + plddt_loss + pae_loss + delta_dis_loss + joint_change_confidence_loss # + distogram_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print("Loss calculation complete")
            
            # loss.backward()

            # print("Backward pass complete")
            # optimizer.step()
            # optimizer.zero_grad()


            processed_feature_dict_detached = tensor_tree_map(
            lambda x: np.array(x[..., -1].detach().cpu()),
            processed_feature_dict
            )
            prev_atom_positions = out["final_atom_positions"].detach().clone()

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
                args["output_dir"], f'{names[target_index]}_{i}_unrelaxed.pdb'
            )

            with open(unrelaxed_output_path, 'w') as fp:
                    fp.write(protein.to_pdb(unrelaxed_protein))

            # NOTE: order of args is important here
            tm_score = run_mmalign_and_get_tmscore(f"references/{names[target_index]}.pdb", unrelaxed_output_path)
            predicted_tm_score = out["weighted_ptm_score"]
            delta_tm = 0
            if(i > 0):
                delta_tm = 1 - run_mmalign_and_get_tmscore(unrelaxed_output_path.replace(f"_{i}_", f"_{i-1}_"), unrelaxed_output_path)

            print(f"Loss: {loss}")
            print(f"Distogram loss: {distogram_loss}")
            print(f"Delta distogram loss: {delta_dis_loss}")
            print(f"plddt: {out['plddt'].mean()}")
            print(f"tm_score: {tm_score}")
            print(f"Weighted ptm score: {weighted_ptm_score}")
            print(f"PAE: {pae}")
            
            # wandb.log({"plddt": out["plddt"].mean(), "tm_score": tm_score, "predicted_tm_score": predicted_tm_score})
            print("Stats logged")
            wandb.log({"plddt": out["plddt"].mean(), "tm_score": tm_score, "predicted_tm_score": predicted_tm_score, "ptm_loss": ptm_loss, "dropout_rate_loss": dropout_rate_loss, "pae_loss": pae_loss, "dropout_binarize_loss": dropout_binarize_loss, "plddt_loss": plddt_loss, "loss": loss, "delta_tm": delta_tm, "delta_dis": delta_dis, "delta_pae": delta_pae, "joint_change_confidence": joint_change_confidence_loss, "pae": pae, "ptm": weighted_ptm_score})

        wandb.finish()