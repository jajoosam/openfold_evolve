import torch
import wandb
import numpy as np
import os

from af_profile_replication import MSAFeatsModule
from dropout_mask import DropoutMaskModule, FapeHistoryLoss

from openfold.utils.script_utils import prep_output
from openfold.np import protein
from openfold.utils.tensor_utils import tensor_tree_map

from utils import (
    run_mmalign_and_get_tmscore,
    get_fape_loss,
)


def run_optimization(num_residues, channels_msa, channels_msa_extra, channels_pair, dropout_rate, processed_feature_dict, iterations, name, seed, model, args, feature_dict, feature_processor, config):
    run = wandb.init(
        project="af_dropout",
        config=config,
    )

    msa_feat_module = MSAFeatsModule(processed_feature_dict["msa_feat"])

    dropout_mask_module = (
        DropoutMaskModule(
            num_residues, channels_msa, channels_msa_extra, channels_pair
        )
        .to(args["model_device"])
        .train()
    )

    fape_history_loss = FapeHistoryLoss(args["model_device"], alpha=run.config["history_alpha"])

    parameters = list(msa_feat_module.parameters()) if config["msa_bias"] else list(dropout_mask_module.parameters())

    optimizer = torch.optim.Adam(
        parameters,
        lr=run.config["learning_rate"]
    )

    reference_pdb = open(f"../references/{name}.pdb").read()
    reference_structure = protein.from_pdb_string(reference_pdb)

    reference_distogram = torch.zeros(num_residues, num_residues)
    for i in range(num_residues):
        for j in range(num_residues):
            reference_distogram[i, j] = torch.norm(
                torch.tensor(reference_structure.atom_positions[i][1])
                - torch.tensor(reference_structure.atom_positions[j][1])
            )

    reference_distogram = reference_distogram.to(args["model_device"])

    fape_history_loss = FapeHistoryLoss(args["model_device"], alpha=run.config["history_alpha"])

    # things to preserve between iterations
    prev_distogram = None
    prev_pae_matrix = None
    prev_atom_positions = None


    # debug NaNs
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(seed)

    # Training loop
    for i in range(iterations):
        optimizer.zero_grad()

        print(f"Iteration #{i} started")

        # add latest masks to feature dict
        current_feature_dict = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v 
                                for k, v in processed_feature_dict.items()}
        masks = {
            "msa_dropout_mask": dropout_mask_module.msa_dropout_mask,  
            "pair_dropout_mask": dropout_mask_module.pair_dropout_mask,
            "extra_msa_dropout_mask": dropout_mask_module.extra_msa_dropout_mask,
        }
        for mask_key in ["msa_dropout_mask", "pair_dropout_mask", "extra_msa_dropout_mask"]:
            current_feature_dict[mask_key] = masks[mask_key]


        # add cluster bias to feature dict
        msa_cluster_bias = msa_feat_module()
        current_feature_dict["msa_feat"][3] += msa_cluster_bias


        out = model(current_feature_dict)
        print(f"Forward pass #{i} complete")


        pae_matrix = out["predicted_aligned_error"]            

        atom_positions = out["final_atom_positions"][:, 1, :]
        distogram = torch.norm(atom_positions[:, None, :] - atom_positions[None, :, :], dim=2)
        distogram = distogram.to(args["model_device"])


        losses = {
            "weighted_ptm_score": -run.config["loss_weights"]["weighted_ptm_score"] * out["weighted_ptm_score"],
            "plddt": -run.config["loss_weights"]["plddt"] * out["plddt"].mean(),
            "pae": run.config["loss_weights"]["pae"] * out["predicted_aligned_error"].mean(),
            "distogram": run.config["loss_weights"]["distogram"] * torch.norm(distogram - reference_distogram),
            "fape_history": run.config["loss_weights"]["fape_history"] * fape_history_loss.compute_loss(out["final_atom_positions"]),
        }

        # joint pae/fape loss
        if i > 0:
            losses["delta_distogram"] = -run.config["loss_weights"]["delta_distogram"] * torch.norm(distogram - prev_distogram)

            fape_loss = get_fape_loss(prev_atom_positions, out["final_atom_positions"]) 

            delta_pae = ((pae_matrix - prev_pae_matrix) / 31.0).abs()

            # normalize
            fape_loss = (fape_loss - fape_loss.mean()) / (fape_loss.std() + 1e-8)
            delta_pae = (delta_pae - delta_pae.mean()) / (delta_pae.std() + 1e-8)
            print(torch.norm(
                    fape_loss -
                    delta_pae,
                ))

            losses["joint_pae_fape"] = run.config["loss_weights"]["joint_pae_fape"] * (
                torch.norm(
                    fape_loss -
                    delta_pae,
                ) + 1e-8
            ).sqrt()

        losses["total"] = sum(losses.values())

        loss = losses["total"]

        print("Loss calculation complete")
        loss.backward()
        print("Backward pass complete")

        optimizer.step()

        fape_history_loss.add_to_history(out["final_atom_positions"], out["weighted_ptm_score"])


        processed_feature_dict_detached = tensor_tree_map(
            lambda x: np.array(x[..., -1].detach().cpu()), processed_feature_dict
        )
        prev_atom_positions = out["final_atom_positions"].detach().clone()
        print("feature_dict: ", processed_feature_dict_detached.keys())
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
            args["output_dir"], f"{name}_{i}_unrelaxed.pdb"
        )

        with open(unrelaxed_output_path, "w") as fp:
            fp.write(protein.to_pdb(unrelaxed_protein))


        tm_score = run_mmalign_and_get_tmscore(f"../references/{name}.pdb", unrelaxed_output_path)
        delta_tm = 1 - run_mmalign_and_get_tmscore(unrelaxed_output_path.replace(f"_{i}_", f"_{i-1}_"), unrelaxed_output_path) if i > 0 else 0

        additional_metrics = {
            "tm_score": tm_score,
            "delta_tm": delta_tm,
            "plddt": out["plddt"].mean().item(),
            "pae": out["predicted_aligned_error"].mean().item(),
            "ptm": out["weighted_ptm_score"].item(),
            "distogram_diff": torch.norm(distogram - reference_distogram),
        }

        wandb.log({**losses, **additional_metrics})

        print("Stats logged")

        prev_distogram = distogram.detach().clone()
        prev_pae_matrix = out["predicted_aligned_error"].detach().clone()
        prev_atom_positions = out["final_atom_positions"].detach().clone()



    wandb.finish()
