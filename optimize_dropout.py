import torch
import torch.nn as nn

from openfold.model.model import AlphaFold
from openfold.config import model_config

from openfold.data.tools import hhsearch, hmmsearch
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.utils.script_utils import prep_output
from openfold.np import protein
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.tensor_utils import tensor_tree_map


import numpy as np
import wandb

import os
import logging

from utils import (
    run_mmalign_and_get_tmscore,
    setup_template_featurizer,
    setup_data_processor,
    precompute_alignments,
    generate_feature_dict,
    get_fape_loss
)


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
    "output_dir": "output",
    "alignment_dir": "alignment_dir",
    "cpus": 28,
    "use_precomputed_alignments": True,
    "model_device": "cuda:0",
}

args["jax_param_path"] = (
    f"openfold/resources/params/v2.2/params_{args['config_preset']}.npz"
)


def get_model_basename(model_path):
    return os.path.splitext(os.path.basename(os.path.normpath(model_path)))[0]


model_basename = get_model_basename(args["jax_param_path"])
model_version = "_".join(model_basename.split("_")[1:])
model = AlphaFold(config)
import_jax_weights_(model, args["jax_param_path"], version=model_version)
model = model.to(args["model_device"])
model = model.train()


data_processor = setup_data_processor(None)
feature_processor = feature_pipeline.FeaturePipeline(config["data"])


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
        "EVQLEESGGGLVQPGGSLRLSCAASGFTFSSYVMSWVRQAPGKGLEWVSDINSGGSRTYYTDSVKGRFTISRDNAKNTLYLQMNSLKPEDTAVYYCARDSLLSTRYLHTSERGQGTQVTVSS",
    ],
    [
        "MSRIITAPHIGIEKLSAISLEELSCGLPDRYALPPDGHPVEPHLERLYPTAQSKRSLWDFASPGYTFHGLHRAQDYRRELDTLQSLLTTSQSSELQAAAALLKCQQDDDRLLQIILNLLHKV",
        "MNITLTKRQQEFLLLNGWLQLQCGHAERACILLDALLTLNPEHLAGRRCRLVALLNNNQGERAEKEAQWLISHDPLQAGNWLCLSRAQQLNGDLDKARHAYQHYLELKDHNESP",
    ],
    [
        "NASINFVATEAHTASAGGAKIIFNTTNNGATGSTEKVVIDQNGNVGVGVGAPTAKMDVNGGIKQPNYGIISAVRNSGGVTASMPWTNAYVLAHQGEMHQWVAGGPILQDSVTGCNAGPDAGVKFDSIATSWGGPYKVIFHTTGSNGAIHLEWSGWQVSLKNSAGTELAIGMGQVFATLHYDPAVSNWRVEHMFGRINNTNFTCW",
        "NASINFVATEAHTASAGGAKIIFNTTNNGATGSTEKVVIDQNGNVGVGVGAPTAKMDVNGGIKQPNYGIISAVRNSGGVTASMPWTNAYVLAHQGEMHQWVAGGPILQDSVTGCNAGPDAGVKFDSIATSWGGPYKVIFHTTGSNGAIHLEWSGWQVSLKNSAGTELAIGMGQVFATLHYDPAVSNWRVEHMFGRINNTNFTCW",
        "NASINFVATEAHTASAGGAKIIFNTTNNGATGSTEKVVIDQNGNVGVGVGAPTAKMDVNGGIKQPNYGIISAVRNSGGVTASMPWTNAYVLAHQGEMHQWVAGGPILQDSVTGCNAGPDAGVKFDSIATSWGGPYKVIFHTTGSNGAIHLEWSGWQVSLKNSAGTELAIGMGQVFATLHYDPAVSNWRVEHMFGRINNTNFTCW",
    ],
]


class DropoutMaskOptimizer(nn.Module):
    def __init__(self, num_residues, channels_msa, channels_msa_extra, channels_pair):
        super(DropoutMaskOptimizer, self).__init__()
        self.msa_dropout_mask = nn.Parameter(
            (torch.rand(num_residues, channels_msa) - 0.5)
        )
        self.pair_dropout_mask = nn.Parameter(
            (torch.rand(4, num_residues, channels_pair) - 0.5)
        )
        self.extra_msa_dropout_mask = nn.Parameter(
            (torch.rand(num_residues, channels_msa_extra) - 0.5)
        )

    def forward(self, noise=0.0, binarized=False):
        masks = {
            "msa_dropout_mask": self.msa_dropout_mask,
            "pair_dropout_mask": self.pair_dropout_mask,
            "extra_msa_dropout_mask": self.extra_msa_dropout_mask,
        }

        # if noise > 0:
        #     for key in masks:
        #         masks[key] = masks[key] + (torch.rand_like(masks[key]) - 0.5) * noise

        # for key in masks:
        #     masks[key] = torch.sigmoid(masks[key])

        # if binarized:
        #     for key in masks:
        #         hard_mask = (masks[key] > 0.5).float()
        #         masks[key] = hard_mask + (hard_mask - masks[key]).detach()

        return masks

    def get_dropout_rates(self, masks):
        return {
            "msa_dropout_rate": 1 - masks["msa_dropout_mask"].mean(),
            "pair_dropout_rate": 1 - masks["pair_dropout_mask"].mean(),
            "extra_msa_dropout_rate": 1 - masks["extra_msa_dropout_mask"].mean(),
        }

    @staticmethod
    def get_binarization_loss(sigmoided_values):
        return -1 * (
            (sigmoided_values * torch.log(sigmoided_values))
            + (1 - sigmoided_values) * torch.log(1 - sigmoided_values)
        ).mean()

    def compute_binarization_loss(self, masks):
        for key in masks:
            masks[key] = torch.sigmoid(masks[key])
        return sum(self.get_binarization_loss(mask) for mask in masks.values())

    def compute_dropout_rate_loss(self, target_rate, masks, binarized=False):
        for key in masks:
            masks[key] = torch.sigmoid(masks[key])
        if(binarized):
            for key in masks:
                hard_mask = (masks[key] > 0.5).float()
                masks[key] = hard_mask + (hard_mask - masks[key]).detach()
        rates = self.get_dropout_rates(masks)
        return sum((rate - target_rate) ** 2 for rate in rates.values())


class DistogramHistoryLoss:
    def __init__(self, device):
        self.history = []
        self.device = device

    def add_to_history(self, distogram, tm_score):
        self.history.append((distogram.detach().clone(), tm_score.detach().clone()))

    def compute_loss(self, current_distogram):
        if not self.history:
            return torch.tensor(0.0, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)

        for past_distogram, tm_score in self.history:
            # Calculate the distance between the current distogram and the past one
            distance = torch.norm(current_distogram - past_distogram)

            # Weight the distance by the inverse of the TM score
            # This encourages moving away from conformations with low TM scores
            weighted_distance = distance * (1 - tm_score)

            # We want to maximize this distance, so we use negative loss
            total_loss += weighted_distance

        # Normalize by the number of historical distograms
        return -(total_loss / len(self.history)).sqrt()


seeds = torch.randint(0, 1000, (3, 10, 3))  # 3 targets, 10 iterations, 3 diff seeds

for target_index in range(1):
    for iteration in range(5):
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

        template_enabled = model.config["template"]["enabled"]

        model.config["template"]["enabled"] = template_enabled and any(
            ["template_" in k for k in processed_feature_dict]
        )

        num_residues = np.array([len(seq) for seq in seqs[target_index]]).sum()
        channels_msa = 256
        channels_msa_extra = 64
        channels_pair = 128
        dropout_rate = 0.15

        torch.set_grad_enabled(True)
        torch.manual_seed(seeds[target_index][iteration][0])

        dropout_mask_optimizer = (
            DropoutMaskOptimizer(
                num_residues, channels_msa, channels_msa_extra, channels_pair
            )
            .to(args["model_device"])
            .train()
        )

        distogram_history_loss = DistogramHistoryLoss(args["model_device"])

        run = wandb.init(
            project="af_dropout",
            config={
                "args": args,
                "loss_weights": {
                    "weighted_ptm_score": 0,  # 1 is ~ 0.5
                    "dropout_rate": 0,  # 1 is ~ 0.33
                    "dropout_binarize": 0,
                    "plddt": 0,
                    "pae": 3,  # 1 is ~15
                    "distogram": 0,  # 1 is ~1000
                    "delta_distogram": 0,  # 1 is ~300
                    "history": 10, # 1 is ~10
                    "joint_pae_fape": 15, # 1 is ~sqrt(500)
                },
                "learning_rate": 8e-4,
                "target": names[target_index] + "_" + str(iteration),
                "recycles": 0,
                "seed_fixed": True,
                "noise": 0,
                "binarized": False,
            },
        )


        optimizer = torch.optim.Adam(
            dropout_mask_optimizer.parameters(), lr=run.config["learning_rate"]
        )

        reference_pdb = open(f"references/{names[target_index]}.pdb").read()
        reference_structure = protein.from_pdb_string(reference_pdb)

        reference_distogram = torch.zeros(num_residues, num_residues)
        for i in range(num_residues):
            for j in range(num_residues):
                reference_distogram[i, j] = torch.norm(
                    torch.tensor(reference_structure.atom_positions[i][1])
                    - torch.tensor(reference_structure.atom_positions[j][1])
                )

        reference_distogram = reference_distogram.to(args["model_device"])


        # things to preserve between iterations
        prev_distogram = None
        prev_pae = None
        prev_pae_matrix = None
        prev_atom_positions = None
        old_masks = {}

        torch.autograd.set_detect_anomaly(True)
        # Training loop
        for i in range(100):
            optimizer.zero_grad()

            masks = dropout_mask_optimizer(noise=run.config["noise"], binarized=run.config["binarized"])

            current_feature_dict = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v 
                                    for k, v in processed_feature_dict.items()}
            
            current_feature_dict["msa_dropout_mask"] = masks["msa_dropout_mask"]
            current_feature_dict["pair_dropout_mask"] = masks["pair_dropout_mask"]
            current_feature_dict["extra_msa_dropout_mask"] = masks["extra_msa_dropout_mask"]


            if i > 0:
                delta_msa_dropout_mask = torch.abs(
                    current_feature_dict["msa_dropout_mask"]
                    - old_masks["msa_dropout_mask"]
                )
                delta_pair_dropout_mask = torch.abs(
                    current_feature_dict["pair_dropout_mask"]
                    - old_masks["pair_dropout_mask"]
                )
                delta_extra_msa_dropout_mask = torch.abs(
                    current_feature_dict["extra_msa_dropout_mask"]
                    - old_masks["extra_msa_dropout_mask"]
                )
                print(
                    "MSA dropout mask change:",
                    delta_msa_dropout_mask.mean(),
                    torch.abs(current_feature_dict["msa_dropout_mask"]).mean(),
                )
                print(
                    "Pair dropout mask change:",
                    delta_pair_dropout_mask.mean(),
                    torch.abs(current_feature_dict["pair_dropout_mask"]).mean(),
                )
                print(
                    "Extra MSA dropout mask change:",
                    delta_extra_msa_dropout_mask.mean(),
                    torch.abs(current_feature_dict["extra_msa_dropout_mask"]).mean(),
                )
            
            old_masks = {
                "msa_dropout_mask": current_feature_dict["msa_dropout_mask"].clone(),
                "pair_dropout_mask": current_feature_dict[
                    "pair_dropout_mask"
                ].clone(),
                "extra_msa_dropout_mask": current_feature_dict[
                    "extra_msa_dropout_mask"
                ].clone(),
            }

            torch.manual_seed(seeds[target_index][iteration][1])
            np.random.seed(seeds[target_index][iteration][2])

            out = model(current_feature_dict)
            print(f"Forward pass #{i} complete")

            weighted_ptm_score = out["weighted_ptm_score"]
            plddt = out["plddt"].mean()

            print((f"original pae shape: { out['predicted_aligned_error'].shape}"))

            pae_matrix = out["predicted_aligned_error"]            
            pae = out["predicted_aligned_error"].mean()

            print(f"Forward pass #{i} complete")

            atom_positions = out["final_atom_positions"][:, 1, :]
            distogram = torch.norm(atom_positions[:, None, :] - atom_positions[None, :, :], dim=2)
            distogram = distogram.to(args["model_device"])

            # Calculate losses
            losses = {
                "weighted_ptm_score": -run.config["loss_weights"]["weighted_ptm_score"] * out["weighted_ptm_score"],
                "dropout_rate": run.config["loss_weights"]["dropout_rate"] * dropout_mask_optimizer.compute_dropout_rate_loss(dropout_rate, masks, binarized=run.config["binarized"]),
                "dropout_binarize": run.config["loss_weights"]["dropout_binarize"] * dropout_mask_optimizer.compute_binarization_loss(masks),
                "plddt": -run.config["loss_weights"]["plddt"] * out["plddt"].mean(),
                "pae": run.config["loss_weights"]["pae"] * out["predicted_aligned_error"].mean(),
                "distogram": run.config["loss_weights"]["distogram"] * torch.norm(distogram - reference_distogram),
                "history": run.config["loss_weights"]["history"] * distogram_history_loss.compute_loss(distogram),
            }

            fape_change = torch.tensor(0)
            pae_change = torch.tensor(0)
            if i > 0:
                losses["delta_distogram"] = -run.config["loss_weights"]["delta_distogram"] * torch.norm(distogram - prev_distogram)

                fape_loss = get_fape_loss(prev_atom_positions, out["final_atom_positions"]) # N X N
                delta_pae = torch.norm(pae_matrix - prev_pae_matrix) / 31.0

                # for logging purposes
                fape_change = fape_loss.clone()
                pae_change = delta_pae.clone()

                # normalize
                fape_loss = fape_loss / fape_loss.mean()
                delta_pae = delta_pae / delta_pae.mean()


                losses["joint_pae_fape"] = run.config["loss_weights"]["joint_pae_fape"] * (
                    torch.norm(
                        fape_loss -
                        delta_pae,
                    )
                ).sqrt()

            losses["total"] = sum(losses.values())

            loss = losses["total"]
            print("Loss calculation complete")

            loss.backward()
            print("Backward pass complete")
            optimizer.step()

            distogram_history_loss.add_to_history(distogram, out["weighted_ptm_score"])


            processed_feature_dict_detached = tensor_tree_map(
                lambda x: np.array(x[..., -1].detach().cpu()), processed_feature_dict
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
                False,
            )

            unrelaxed_output_path = os.path.join(
                args["output_dir"], f"{names[target_index]}_{i}_unrelaxed.pdb"
            )

            with open(unrelaxed_output_path, "w") as fp:
                fp.write(protein.to_pdb(unrelaxed_protein))


            tm_score = run_mmalign_and_get_tmscore(f"references/{names[target_index]}.pdb", unrelaxed_output_path)
            delta_tm = 1 - run_mmalign_and_get_tmscore(unrelaxed_output_path.replace(f"_{i}_", f"_{i-1}_"), unrelaxed_output_path) if i > 0 else 0

            additional_metrics = {
                "tm_score": tm_score,
                "delta_tm": delta_tm,
                "plddt": out["plddt"].mean().item(),
                "pae": out["predicted_aligned_error"].mean().item(),
                "ptm": out["weighted_ptm_score"].item(),
                "fape_change": fape_change.sum().item(),
                "pae_change": pae_change.sum().item(),
            }

            # Log all metrics
            wandb.log({**losses, **additional_metrics})

            print("Stats logged")

            # Update previous data for next iteration
            prev_distogram = distogram.detach().clone()
            prev_pae = out["predicted_aligned_error"].mean().detach().clone()
            prev_pae_matrix = out["predicted_aligned_error"].detach().clone()
            prev_atom_positions = out["final_atom_positions"].detach().clone()



        wandb.finish()
