import subprocess
import re
import torch

import os
import logging
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.data.tools import hhsearch, hmmsearch

def run_mmalign_and_get_tmscore(reference, model):
    # Define the command
    command = ["./../../USalign/MMalign", reference, model]
    
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




logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

def setup_template_searcher(args):
    if "multimer" in args["config_preset"]:
        return hmmsearch.Hmmsearch(
            binary_path=args["hmmsearch_binary_path"],
            hmmbuild_binary_path=args["hmmbuild_binary_path"],
            database_path=args["pdb_seqres_database_path"],
        )
    else:
        return hhsearch.HHSearch(
            binary_path=args["hhsearch_binary_path"],
            databases=[args["pdb70_database_path"]],
        )

def setup_template_featurizer(args):
    return templates.HmmsearchHitFeaturizer(
        mmcif_dir=args["template_mmcif_dir"],
        max_template_date=args["max_template_date"],
        max_hits=args["max_hits"],
        kalign_binary_path=args["kalign_binary_path"],
    )

def setup_alignment_runner(args, template_searcher):
    return data_pipeline.AlignmentRunner(
        jackhmmer_binary_path=args["jackhmmer_binary_path"],
        hhblits_binary_path=args["hhblits_binary_path"],
        uniref90_database_path=args["uniref90_database_path"],
        mgnify_database_path=args["mgnify_database_path"],
        bfd_database_path=args["bfd_database_path"],
        uniref30_database_path=args["uniref30_database_path"],
        uniprot_database_path=args["uniprot_database_path"],
        template_searcher=template_searcher,
        use_small_bfd=args["bfd_database_path"] is None,
        no_cpus=args["cpus"],
    )

def setup_data_processor(template_featurizer):
    monomer_data_pipeline = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )
    return data_pipeline.DataPipelineMultimer(
        monomer_data_pipeline=monomer_data_pipeline,
    )

def precompute_alignments(tags, seqs, alignment_dir, args):
    for tag, seq in zip(tags, seqs):
        tmp_fasta_path = os.path.join(args["output_dir"], f"tmp_{os.getpid()}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)

        if not args["use_precomputed_alignments"]:
            logger.info(f"Generating alignments for {tag}...")
            os.makedirs(local_alignment_dir, exist_ok=True)
            
            template_searcher = setup_template_searcher(args)
            alignment_runner = setup_alignment_runner(args, template_searcher)
            alignment_runner.run(tmp_fasta_path, local_alignment_dir)
            
            logger.info("Alignment done!")
        else:
            logger.info(f"Using precomputed alignments for {tag} at {alignment_dir}...")

        os.remove(tmp_fasta_path)

def generate_feature_dict(tags, seqs, alignment_dir, data_processor, args):
    tmp_fasta_path = os.path.join(args["output_dir"], f"tmp_{os.getpid()}.fasta")

    if "multimer" in args["config_preset"]:
        with open(tmp_fasta_path, "w") as fp:
            fp.write("\n".join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)]))
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path,
            alignment_dir=alignment_dir,
        )
    elif len(seqs) == 1:
        tag, seq = tags[0], seqs[0]
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
            fp.write("\n".join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)]))
        feature_dict = data_processor.process_multiseq_fasta(
            fasta_path=tmp_fasta_path,
            super_alignment_dir=alignment_dir,
        )

    os.remove(tmp_fasta_path)
    return feature_dict

atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}


def get_fape_loss(inputs, outputs, copies=1, clamp=31.0):
    def robust_norm(x, dim=-1, keepdim=False, eps=1e-8):
        return torch.sqrt(torch.sum(x ** 2, dim=dim, keepdim=keepdim) + eps)
    
    def get_R(N, CA, C):
        v1, v2 = C - CA, N - CA
        e1 = v1 / robust_norm(v1, dim=-1, keepdim=True)
        c = torch.einsum('li,li->l', e1, v2).unsqueeze(-1)
        e2 = v2 - c * e1
        e2 = e2 / robust_norm(e2, dim=-1, keepdim=True)
        e3 = torch.cross(e1, e2, dim=-1)
        return torch.cat([e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1)
    
    def get_ij(R, T):
        return torch.einsum('rji,rsj->rsi', R, T[None, :] - T[:, None])
    
    def loss_fn(t, p):
        fape = robust_norm(t - p)
        fape = torch.clamp(fape, 0, clamp) / clamp
        return fape
    
    true = inputs
    pred = outputs
    N, CA, C = (atom_order[k] for k in ["N", "CA", "C"])
    
    
    true = get_ij(get_R(true[:, N], true[:, CA], true[:, C]), true[:, CA])
    pred = get_ij(get_R(pred[:, N], pred[:, CA], pred[:, C]), pred[:, CA])
    
    fape = loss_fn(true, pred)
    return fape

def make_hard_mask(mask_logits, threshold=0.15):

    mask_probs = torch.softmax(mask_logits.view(-1), dim=-1)
    sorted_mask, _ = torch.sort(mask_probs, descending=False)
    threshold_index = int(threshold * mask_logits.numel())
    threshold = sorted_mask[threshold_index]

    hard_mask = (mask_probs > threshold).float()
    
    hard_mask = hard_mask.view(mask_logits.shape)

    return hard_mask