#!/usr/bin/env python
"""Plot final comparison across all methods"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load results
methods_data = [
    ("Baseline", "final_baseline.json", "black", "-"),
    ("QFilter", "final_qfilter.json", "gray", "--"),
    ("KVMerger (cosine)", "final_kvmerger_cosine.json", "red", "-."),
    ("KVMerger (L2)", "final_kvmerger_l2.json", "orange", ":"),
    ("LSE (ε=35)", "final_lse.json", "blue", "-"),
    ("LSE (ε=40, r=16)", "final_lse_r16.json", "green", "--"),
]

problems = [6, 12, 18, 24, 30]

fig, ax = plt.subplots(figsize=(12, 6))

for name, filename, color, linestyle in methods_data:
    path = Path(f"analysis/{filename}")
    if path.exists():
        data = json.load(open(path))
        ppls = data['perplexities']
        ax.plot(problems, ppls, marker='o', label=name, color=color, linestyle=linestyle, linewidth=2)

ax.set_xlabel('AIME Problem Number', fontsize=12)
ax.set_ylabel('Perplexity (lower = better)', fontsize=12)
ax.set_title('KV Cache Compression Methods - Perplexity Comparison\n(5 AIME problems, 256 tokens each)', fontsize=14)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(problems)

plt.tight_layout()
plt.savefig('analysis/final_comparison_plot.png', dpi=150, bbox_inches='tight')
print("Saved to analysis/final_comparison_plot.png")

# Also create a bar chart of average perplexities
fig2, ax2 = plt.subplots(figsize=(10, 6))

methods_avg = []
for name, filename, color, _ in methods_data:
    path = Path(f"analysis/{filename}")
    if path.exists():
        data = json.load(open(path))
        methods_avg.append((name, data['avg_perplexity'], color))

methods_avg.sort(key=lambda x: x[1])
names = [m[0] for m in methods_avg]
avgs = [m[1] for m in methods_avg]
colors = [m[2] for m in methods_avg]

bars = ax2.barh(names, avgs, color=colors, alpha=0.7)
ax2.set_xlabel('Average Perplexity (lower = better)', fontsize=12)
ax2.set_title('Average Perplexity by Method', fontsize=14)
ax2.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, avgs)):
    ax2.text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('analysis/final_comparison_avg.png', dpi=150, bbox_inches='tight')
print("Saved to analysis/final_comparison_avg.png")

print("\nDone!")
