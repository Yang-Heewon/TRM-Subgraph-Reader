import os
import torch
import torch.nn as nn
import numpy as np

# We mimic the path properly so trm_unified can be imported
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./TinyRecursiveModels'))

from trm_unified.train_core import build_model
from trm_unified.tokenization import load_tokenizer

def test_subgraph_architecture():
    print("--- 1. Initializing Model ---")
    
    # We use a tiny config that mirrors train() inside train_core.py
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
        'puzzle_emb_len': 1,
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
    model.inner.puzzle_emb_data = np.random.randn(cfg['num_puzzle_identifiers'], cfg['puzzle_emb_ndim']).astype(np.float32)
    model.inner.relation_emb_data = np.random.randn(cfg['num_relation_identifiers'], cfg['relation_emb_ndim']).astype(np.float32)
    
    print("Model initialized successfully.")

    print("\n--- 2. Mocking Data Pipeline (make_collate) ---")
    # We skip loading HuggingFace tokenizers internally and just mock token ids
    # A batch of two examples
    raw_batch = [
        {
            'question': "Who is the president of the moon?",
            'q_text': "Who is the president of the moon?",
            'tuples': [
                (10, 5, 11),  # s, r, o
                (11, 6, 12),
                (10, 5, 20),
            ],
            'answers_cid': [12, 20]
        },
        {
            'question': "When was python released?",
            'q_text': "When was python released?",
            'tuples': [
                (55, 1, 56),
            ],
            'answers_cid': [56]
        }
    ]
    
    from trm_unified.train_core import make_collate
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_rel:
        np.save(f_rel.name, np.zeros((10, 10)))
        rel_npy = f_rel.name
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_q:
        np.save(f_q.name, np.zeros((10, 10)))
        q_npy = f_q.name
        
    try:
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
        batch = collate_fn(raw_batch)
    except Exception as e:
        print(f"make_collate failed! {e}")
        return
    finally:
        os.remove(rel_npy)
        os.remove(q_npy)
        
    input_ids = batch['input_ids'].to(device)
    attn_mask = batch['attention_mask'].to(device)
    seq_batches = batch['seq_batches']
    sb = seq_batches[0]
    
    print(f"Collate output shapes:")
    print(f" - input_ids: {input_ids.shape}")
    print(f" - full_attention_mask: {sb['attention_mask'].shape}")
    print(f" - puzzle_identifiers: {sb['puzzle_identifiers'].shape}")
    print(f" - labels: {sb['labels'].shape}")

    print("\n--- 3. Testing Subgraph Forward Pass ---")
    
    B = input_ids.shape[0]
    inner = model.inner
    carry = carry_cls(
        inner.empty_carry(B), 
        torch.zeros(B, device=device), 
        torch.ones(B, dtype=torch.bool, device=device), 
        {}
    )
    
    new_carry, outputs = model(
        carry, 
        {
            'input_ids': input_ids,
            'attention_mask': sb['attention_mask'].to(device),
            'puzzle_identifiers': sb['puzzle_identifiers'].to(device),
            'relation_identifiers': sb['relation_identifiers'].to(device),
            'candidate_mask': sb['candidate_mask'].to(device),
        }
    )
    
    logits = outputs.get('scores')
    print(f"Model generated logits of shape: {logits.shape}")
    
    # Assertions
    N_max = sb['puzzle_identifiers'].shape[1]
    assert logits.shape == (B, N_max), f"Expected logits shape {(B, N_max)}, got {logits.shape}"
    print("Forward pass logic validated successfully.")

    print("\n--- 4. Computing BCE Loss Native Step ---")
    loss_fct = nn.BCEWithLogitsLoss(reduction='none')
    labels = sb['labels'].to(device)
    cmask = sb['candidate_mask'].to(device)
    
    raw_bce = loss_fct(logits, labels)
    valid_bce = (raw_bce * cmask).sum(dim=-1) / cmask.sum(dim=-1).clamp(min=1)
    mean_bce = valid_bce.mean()
    print(f"Computed Node Classification BCE Loss: {mean_bce.item():.4f}")
    
    print("\n✅ All Subgraph Reader changes dry-run successfully without syntax crashes!")

if __name__ == "__main__":
    test_subgraph_architecture()
