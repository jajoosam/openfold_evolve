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
    run_tmalign_and_get_tmscore
)

from lookahead import Lookahead

import esm 

if_model, if_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

# use eval mode for deterministic output e.g. without random dropout
if_model = if_model.eval().cuda()

def run_optimization(num_residues, channels_msa, channels_msa_extra, channels_pair, dropout_rate, processed_feature_dict, iterations, name, seed, model, args, feature_dict, feature_processor, config, chains):
    print(processed_feature_dict.keys())
    if(config["soloseq"]):
        soloseq = True
    else:
        soloseq = False
    run = wandb.init(
        project="af_dropout",
        config=config,
    )

    model.train()

    # for param in model.parameters():
    #     param.requires_grad = False

    # msa_feat_module = MSAFeatsModule(processed_feature_dict["msa_feat"])

    dropout_mask_module = (
        DropoutMaskModule(
            num_residues, channels_msa, channels_msa_extra, channels_pair, init=config["init"]
        )
        .to(args["model_device"])
        .train()
    )

    fape_history_loss = FapeHistoryLoss(args["model_device"], alpha=run.config["history_alpha"])

    # parameters = list(msa_feat_module.parameters()) if config["msa_bias"] else list(dropout_mask_module.parameters())
    parameters = list(dropout_mask_module.parameters())
    print(dropout_mask_module.msa_dropout_mask.shape)

    optimizer = torch.optim.Adam(
        parameters,
        lr=run.config["learning_rate"]
    )
    # optimizer = Lookahead(optimizer, alpha=0.5, k=5)


    reference_pdb = open(f"../references/{name.split('_')[0]}.pdb").read()
    reference_structure = protein.from_pdb_string(reference_pdb)

    reference_distogram = torch.zeros(num_residues, num_residues)
    for i in range(num_residues):
        for j in range(num_residues):
            reference_distogram[i, j] = torch.norm(
                torch.tensor(reference_structure.atom_positions[i][1])
                - torch.tensor(reference_structure.atom_positions[j][1])
            )

    reference_distogram = reference_distogram.to(args["model_device"])


    # debug NaNs
    # torch.autograd.set_detect_anomaly(True)
    
    if(seed is not None):
        print(f"Seed: {seed}")
        torch.manual_seed(seed)

    prev_pae = float('inf')

    # Training loop
    for i in range(iterations):
        optimizer.zero_grad()

        print(f"Iteration #{i} started")
        torch.manual_seed(seed)

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
            print(f"Len {mask_key}: {masks[mask_key].numel()}")
            print(f"Dropout std {mask_key}: {torch.std((masks[mask_key]))}")
            print(f"Dropout mean {mask_key}: {torch.mean((masks[mask_key]))}")


        # add cluster bias to feature dict
        # msa_cluster_bias = msa_feat_module()
        # current_feature_dict["msa_feat"][3] += msa_cluster_bias


        out = model(current_feature_dict)
        print(f"Forward pass #{i} complete")



        atom_positions = out["final_atom_positions"][:, 1, :]
        distogram = torch.norm(atom_positions[:, None, :] - atom_positions[None, :, :], dim=2)
        distogram = distogram.to(args["model_device"])

        # if
        coords = {

        }
        seqs = {}

        alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        for chain_idx in range(len(chains)):
            chain_id = alphabets[chain_idx]

            chain_start = chains[chain_idx]["start"]
            chain_end = chains[chain_idx]["end"]

            coords[chain_id] = torch.zeros(len(chains[chain_idx]["seq"]), 3 , 3).to(args["model_device"])
            coords[chain_id][:, 0, :] = out["final_atom_positions"][:, 0, :][chain_start:chain_end]
            coords[chain_id][:, 1, :] = out["final_atom_positions"][:, 1, :][chain_start:chain_end]
            coords[chain_id][:, 2, :] = out["final_atom_positions"][:, 2, :][chain_start:chain_end]

            seqs[chain_id] = chains[chain_idx]["seq"]
        
        ll_score = 0
        seq_len = 0
        chain_idx = np.random.randint(0, len(chains))
        with torch.enable_grad():
            for chain_idx in range(len(chains)):
                target_chain_id = alphabets[chain_idx]
                target_seq = seqs[target_chain_id]
                _, ll = esm.inverse_folding.multichain_util.score_sequence_in_complex(
                    if_model, if_alphabet, coords, target_chain_id, target_seq, padding_length=10)
                ll_score += ll * len(target_seq)
                seq_len += len(target_seq)
            
            ll_score = ll_score / seq_len

        print(f"nll {-ll_score}")    
        print(out.keys())
        losses = {
            "weighted_ptm_score": run.config["loss_weights"]["weighted_ptm_score"] * (1/out["ptm_score" if soloseq else "weighted_ptm_score"]),
            "distogram": run.config["loss_weights"]["distogram"] * torch.norm(distogram - reference_distogram),
            "fape_history": run.config["loss_weights"]["fape_history"] * fape_history_loss.compute_loss(out["final_atom_positions"]),
            "nll": run.config["loss_weights"]["nll"] * -ll_score,
            "pae": run.config["loss_weights"]["pae"] * out["predicted_aligned_error"].mean(),
            # "nll_ptm_mul": run.config["loss_weights"]["nll_ptm_mul"] * (1/out["weighted_ptm_score"]) * -ll_score,
        }


        losses["total"] = sum(losses.values())

        loss = losses["total"]
        print("Loss calculation complete")
        
        # current_pae = out["predicted_aligned_error"].mean().item()
        # if(current_pae > prev_pae):
        #     continue
        # prev_pae = current_pae

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        
        optimizer.step()

        # fape_history_loss.add_to_history(out["final_atom_positions"], out["weighted_ptm_score"])

        processed_feature_dict_detached = tensor_tree_map(
            lambda x: np.array(x[..., -1].detach().cpu()), processed_feature_dict
        )

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
    
        score_functon = run_tmalign_and_get_tmscore if config["soloseq"] else run_mmalign_and_get_tmscore

        tm_score = score_functon(f"../references/{name.split('_')[0]}.pdb", unrelaxed_output_path)
        delta_tm = 1 - score_functon(unrelaxed_output_path.replace(f"_{i}_", f"_{i-1}_"), unrelaxed_output_path) if i > 0 else 0
        print(f"TM score {tm_score}")
        additional_metrics = {
            "tm_score": tm_score,
            "delta_tm": delta_tm,
            "plddt": out["plddt"].mean().item(),
            "pae": out["predicted_aligned_error"].mean().item(),
            # "ptm": out["weighted_ptm_score"].item(),
            "distogram_diff": torch.norm(distogram - reference_distogram),
            "nll": -ll_score,
        }

        wandb.log({**losses, **additional_metrics})

        print("Stats logged")



    wandb.finish()
