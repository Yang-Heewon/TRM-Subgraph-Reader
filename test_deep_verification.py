import os
import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./TinyRecursiveModels'))

from trm_unified.train_core import build_model
from trm_unified.tokenization import load_tokenizer

def deep_test_subgraph_architecture():
    print("=== Deep Verification of Subgraph Reader ===")
    
    cfg = {
        'batch_size': 2,
        'seq_len': 32,
        'vocab_size': 30000,
        'hidden_size': 64,
        'num_heads': 2,
        'expansion': 4,
        'H_cycles': 1,
        'L_cycles': 0,
        'L_layers': 2,
        'H_layers': 0,
        'puzzle_emb_ndim': 32,
        'relation_emb_ndim': 32,
        'puzzle_emb_len': 1, # will be dynamic based on batch
        'pos_encodings': 'none',
        'forward_dtype': 'float32',
        'halt_max_steps': 4,
        'halt_exploration_prob': 0.1,
        'no_ACT_continue': True,
        'mlp_t': False,
        'num_puzzle_identifiers': 2000,
        'num_relation_identifiers': 200,
    }
    
    trm_root = os.path.abspath('./TinyRecursiveModels')
    model, carry_cls = build_model('trm_hier6', trm_root, cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval() # Prevent dropout for deterministic tests
    
    model.inner.puzzle_emb_data = np.random.randn(cfg['num_puzzle_identifiers'], cfg['puzzle_emb_ndim']).astype(np.float32)
    model.inner.relation_emb_data = np.random.randn(cfg['num_relation_identifiers'], cfg['relation_emb_ndim']).astype(np.float32)

    from trm_unified.train_core import make_collate, _parse_eval_example
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_rel:
        np.save(f_rel.name, np.zeros((10, 10)))
        rel_npy = f_rel.name
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_q:
        np.save(f_q.name, np.zeros((10, 10)))
        q_npy = f_q.name

    collate_fn = make_collate(
        tokenizer_name='bert-base-uncased',
        rel_npy=rel_npy,
        q_npy=q_npy,
        max_neighbors=256,
        prune_keep=64,
        prune_rand=64,
        max_q_len=16,
        max_steps=4,
        entity_vocab_size=cfg['num_puzzle_identifiers']
    )

    # Batch with two examples, different number of nodes
    # Ex 1: 4 nodes. Ex 2: 2 nodes.
    raw_batch = [
        {
            'question': "Who is the president of the moon?",
            'q_text': "Who is the president of the moon?", # 8 tokens approx
            'tuples': [
                (10, 5, 11),  
                (11, 6, 12),
                (10, 5, 20),
            ],
            'answers_cid': [12, 20]
        },
        {
            'question': "Very short list?",
            'q_text': "Very short list?", # 4 tokens approx
            'tuples': [
                (55, 1, 56),
            ],
            'answers_cid': [56]
        }
    ]
    
    batch = collate_fn(raw_batch)
    
    input_ids = batch['input_ids'].to(device)
    attn_mask = batch['attention_mask'] # Text query mask
    sb = batch['seq_batches'][0]
    full_am = sb['attention_mask'].to(device)
    
    B = input_ids.shape[0]
    Q_len = input_ids.shape[1]
    N_max = sb['puzzle_identifiers'].shape[1]
    L_total = Q_len + N_max
    
    print(f"\n[Validation 1] Attention Mask Shapes:")
    print(f"Max Nodes (N_max): {N_max}")
    print(f"Query Tokens (Q_len): {Q_len}")
    print(f"Expected Full Mask Shape: ({B}, 1, {L_total}, {L_total})")
    print(f"Actual Full Mask Shape: {full_am.shape}")
    assert full_am.shape == (B, 1, L_total, L_total), "Full attention mask shape mismatch"

    # Deep check logic of mask structure
    print(f"\n[Validation 2] Checking Masking Logic (Graph Connectivity):")
    # For ex 0, it has 4 nodes. Indices 0,1,2,3 are nodes.
    # Nodes can attend to each other if they have edge, AND can they attend to query?
    q_mask = attn_mask[0]
    valid_q_len = int(q_mask.sum())
    
    # Are nodes allowed to attend to the query?
    # Inspecting mask[0, 0, 0:4 (nodes), N_max:N_max+valid_q_len (queries)]
    node_to_query_mask = full_am[0, 0, 0:4, N_max:N_max+valid_q_len]
    print(f"Node-to-Query mask (all True?): {node_to_query_mask.all().item()}")
    
    # Check padded nodes (index 2-3 in ex 1, since only 2 nodes)
    # They should be completely blocked
    padded_node_mask = full_am[1, 0, :, 2:N_max]
    print(f"Padded node column blocked completely? {(padded_node_mask == False).all().item()}")

    print(f"\n[Validation 3] Model Logit Output:")
    inner = model.inner
    carry = carry_cls(
        inner.empty_carry(B), 
        torch.zeros(B, device=device), 
        torch.ones(B, dtype=torch.bool, device=device), 
        {}
    )
    
    _, outputs = model(
        carry, 
        {
            'input_ids': input_ids,
            'attention_mask': full_am,
            'puzzle_identifiers': sb['puzzle_identifiers'].to(device),
            'relation_identifiers': sb['relation_identifiers'].to(device),
            'candidate_mask': sb['candidate_mask'].to(device),
        }
    )
    
    logits = outputs.get('scores')
    print(f"Extracted node logits shape: {logits.shape}")
    assert logits.shape == (B, N_max), "Logits should only contain scores for Nodes, not queries"

    print(f"\n[Validation 4] Metric Computation Logic:")
    labels = sb['labels'].to(device)
    cmask = sb['candidate_mask'].to(device)
    
    # Apply cmask with -inf for proper TopK
    masked_logits = logits.clone()
    masked_logits[~cmask] = float('-inf')
    
    # Prediction
    predicted = (masked_logits > 0).float()
    
    # F1 Calculation (mimicking evaluate_relation_beam)
    intersection = (predicted * labels).sum(dim=-1)
    union = predicted.sum(dim=-1) + labels.sum(dim=-1)
    f1 = torch.where(union > 0, 2.0 * intersection / union, torch.tensor(0.0, device=device))
    print(f"F1 scores per instance: {f1.tolist()}")
    
    # Hit@1 Calculation
    top1_idx = masked_logits.argmax(dim=-1)
    hit1 = labels[torch.arange(B), top1_idx]
    
    # Edge case: If an example has no gold labels, hit1 should be false or excluded.
    print(f"Hit@1 per instance: {hit1.tolist()}")

    print("\n✅ Verification Test Passed Successfully!")

    os.remove(rel_npy)
    os.remove(q_npy)

if __name__ == "__main__":
    deep_test_subgraph_architecture()
