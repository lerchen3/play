Various ideas with LLM specialization, especially for computational math questions.

**Knowledge distillation through hinting** (`data/wlogprobs.py`, `data/hint_gemini.py`, `data/hint_openai.py`):
Smarter closed-source model identifies exact failure points and scaffolds correction through continuation.

**Synthetic Data Generation** (`data/anchored.py`):
Uses o3-mini to create fundamentally different but equivalent math problems. Generates denser training distributions.

**Natural Language Reward Modeling** (`judge/grpo_nl.py`):
GRPO-trained reward models that judge solution correctness in natural language. Used in conjunction with various search algorithms.

**Autoformalization** (`autoformalization_training/`):
Converts natural language mathematics to Lean4 formal proofs using vLLM. Implements comparative verification between statements.

**Competition Infrastructure** (`eval/`):
Evaluation on a held out test set.

*Spring 2025 - Experimental approaches to LLM mathematical reasoning*