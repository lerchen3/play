#!/usr/bin/env python
"""Plot compression ratio vs perplexity for all methods"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# Collect all results
all_results = []

# Pattern: analysis/final_*.json
for result_file in sorted(glob("analysis/final_*.json")):
    data = json.load(open(result_file))
    
    method = data.get('method', 'unknown')
    epsilon = data.get('epsilon')
    rank = data.get('rank')
    avg_ppl = data['avg_perplexity']
    
    # Estimate compression from method
    # These are rough estimates - ideally we'd store compression in the JSON
    if method == "baseline":
        compression = 100.0
        label = "Baseline"
    elif method == "qfilter":
        compression = 50.0  # Rough estimate for max_length=128
        label = "QFilter"
    elif method == "kvmerger_cosine":
        compression = 75.0  # From earlier tests
        label = "KVMerger (cos)"
    elif method == "kvmerger_l2":
        compression = 80.0  # From earlier tests
        label = "KVMerger (L2)"
    elif method == "lse":
        # Estimate based on epsilon (from our 256-token tests)
        if epsilon == 20:
            compression = 16.0
        elif epsilon == 25:
            compression = 12.0
        elif epsilon == 30:
            compression = 11.0
        elif epsilon == 35:
            compression = 10.5
        elif epsilon == 40:
            compression = 10.2
        elif epsilon == 45:
            compression = 10.0
        else:
            compression = 10.0
        
        if rank:
            label = f"LSE ε={int(epsilon)} r={rank}"
        else:
            label = f"LSE ε={int(epsilon)}"
    else:
        continue
    
    all_results.append({
        'method': method,
        'epsilon': epsilon,
        'rank': rank,
        'compression': compression,
        'perplexity': avg_ppl,
        'label': label,
    })

# Sort by compression
all_results.sort(key=lambda x: x['compression'])

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))

# Separate by method type for coloring
colors = {
    'baseline': 'black',
    'qfilter': 'gray',
    'kvmerger_cosine': 'red',
    'kvmerger_l2': 'orange',
    'lse': 'blue',
}

markers = {
    'baseline': 's',
    'qfilter': '^',
    'kvmerger_cosine': 'o',
    'kvmerger_l2': 'D',
    'lse': 'o',
}

for r in all_results:
    method = r['method']
    has_rank = r['rank'] is not None
    
    color = colors.get(method, 'blue')
    marker = markers.get(method, 'o')
    
    # Rank projections use different marker
    if has_rank:
        marker = 'x'
        color = 'green'
    
    ax.scatter(r['compression'], r['perplexity'], 
              color=color, marker=marker, s=100, alpha=0.7,
              label=r['label'] if r['label'] not in [x.get_label() for x in ax.get_children()] else "")
    
    # Add label
    ax.annotate(r['label'], (r['compression'], r['perplexity']),
               xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

ax.set_xlabel('Compression % (higher = less compression)', fontsize=12)
ax.set_ylabel('Perplexity (lower = better)', fontsize=12)
ax.set_title('KV Cache Compression: Compression vs Perplexity Trade-off', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 105)

# Add reference line at baseline
baseline_ppl = next((r['perplexity'] for r in all_results if r['method'] == 'baseline'), None)
if baseline_ppl:
    ax.axhline(y=baseline_ppl, color='black', linestyle='--', alpha=0.3, label='Baseline PPL')

plt.tight_layout()
plt.savefig('analysis/compression_vs_perplexity.png', dpi=150, bbox_inches='tight')
print("Saved to analysis/compression_vs_perplexity.png")

# Print summary table
print("\n" + "=" * 80)
print("Compression vs Perplexity Summary")
print("=" * 80)
print(f"{'Method':<30} | {'Compression':<12} | {'Perplexity':<12}")
print("-" * 80)
for r in sorted(all_results, key=lambda x: x['perplexity']):
    print(f"{r['label']:<30} | {r['compression']:>10.1f}% | {r['perplexity']:>12.4f}")
print("=" * 80)

