import argparse
import os
import sys
from typing import Tuple, Optional

import torch
from transformers import AutoTokenizer

# Add the project root to Python path for all imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Also add current directory for local imports
sys.path.insert(0, os.path.dirname(__file__))

from train.model import DeepSeekV3Model
from specdec import generate_speculative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with DeepSeek-V3 using latent c_kv cache (recompute K,V).")

    # Required inputs
    parser.add_argument('--model_save_path', type=str, required=True,
                        help='Path to .pt weights (model_state_dict)')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='HF tokenizer directory used for training (e.g., Qwen path)')

    # Prompt and decoding
    parser.add_argument('--prompt', type=str, default='', help='User prompt text')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Max new tokens to generate')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling (0 disables)')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p nucleus sampling')
    parser.add_argument('--temperature', type=float, default=1.0, help='Softmax temperature')
    parser.add_argument('--use_specdec', action='store_true', help='Enable speculative decoding using MTP')
    parser.add_argument('--mtp_depth', type=int, default=0, help='Depth of MTP block for speculation (0 disables)')

    # Device
    parser.add_argument('--device', type=str, default='cuda:0', help="Torch device, e.g., 'cpu' or 'cuda:0'")

    # Model hyperparameters (must match training). Defaults mirror train/train.py and match provided CLI
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dc_kv', type=int, default=16)
    parser.add_argument('--dc_q', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--rmsnorm_eps', type=float, default=1e-6)
    parser.add_argument('--n_shared_experts', type=int, default=1)
    parser.add_argument('--n_routed_experts', type=int, default=2)
    parser.add_argument('--k_routed_experts', type=int, default=1)
    parser.add_argument('--d_ff_expert_mult', type=int, default=2)
    parser.add_argument('--moe_balance_factor', type=float, default=0.01)
    parser.add_argument('--bias_update_speed', type=float, default=0.001)
    parser.add_argument('--d_head', type=int, default=16, help='Override attention head dim (default d_model/n_heads)')

    return parser.parse_args()


def build_args_from_tokenizer(cli_args: argparse.Namespace, tokenizer) -> argparse.Namespace:
    # Build a lightweight args object for model init
    args = argparse.Namespace()
    args.vocab_size = len(tokenizer)
    args.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Copy model hyperparameters from CLI
    args.seq_len = cli_args.seq_len
    args.d_model = cli_args.d_model
    args.n_heads = cli_args.n_heads
    args.dc_kv = cli_args.dc_kv
    args.dc_q = cli_args.dc_q
    args.num_layers = cli_args.num_layers
    args.rmsnorm_eps = cli_args.rmsnorm_eps
    args.n_shared_experts = cli_args.n_shared_experts
    args.n_routed_experts = cli_args.n_routed_experts
    args.k_routed_experts = cli_args.k_routed_experts
    args.d_ff_expert_mult = cli_args.d_ff_expert_mult
    args.d_ff_expert = args.d_model * args.d_ff_expert_mult
    args.moe_balance_factor = cli_args.moe_balance_factor
    args.bias_update_speed = cli_args.bias_update_speed
    args.mtp_depth = cli_args.mtp_depth
    args.device = torch.device(cli_args.device)
    args.d_head = cli_args.d_head
    return args


def _drop_rope_buffers_for_seq_change(state_dict: dict) -> dict:
    keys_to_drop = [
        k for k in list(state_dict.keys())
        if (
            '.attn.rope_k.cos' in k or '.attn.rope_k.sin' in k or
            '.attn.rope_q.cos' in k or '.attn.rope_q.sin' in k
        )
    ]
    for k in keys_to_drop:
        state_dict.pop(k, None)
    return state_dict


def load_model(weights_path: str, args_for_model: argparse.Namespace, device: torch.device) -> DeepSeekV3Model:
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"No such file: {weights_path}")

    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)

    state_dict = _drop_rope_buffers_for_seq_change(state_dict)

    model = DeepSeekV3Model(args_for_model).to(device)
    
    # Load with strict=False to handle potential parameter mismatches
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys when loading model: {missing_keys[:5]}...")  # Show first 5 only
    if unexpected_keys:
        print(f"Warning: Unexpected keys when loading model: {unexpected_keys[:5]}...")  # Show first 5 only
    
    model.eval()
    
    # Set model to be more numerically stable
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Disable gradient computation completely for inference
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def _top_k_top_p_filtering(probs: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k and top_k > 0:
        values, indices = torch.topk(probs, min(top_k, probs.numel()))
        mask = torch.zeros_like(probs)
        mask.scatter_(0, indices, values)
        probs = mask
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        keep = cumulative <= top_p
        if keep.numel() > 0:
            keep[0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(0, sorted_indices[keep], sorted_probs[keep])
        probs = filtered
    return probs


def _sample_from_logits(logits: torch.Tensor, top_k: int, top_p: float, temperature: float, vocab_size: int) -> int:
    # Ensure logits are finite and have correct size
    if not torch.isfinite(logits).all():
        print("Warning: Non-finite values in logits. Replacing with zeros.")
        logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
    
    # Ensure we have the right vocab size
    if logits.shape[0] != vocab_size:
        print(f"Warning: Logits shape {logits.shape[0]} != vocab_size {vocab_size}")
        if logits.shape[0] > vocab_size:
            logits = logits[:vocab_size]
        else:
            padding = torch.full((vocab_size - logits.shape[0],), -1e9, device=logits.device, dtype=logits.dtype)
            logits = torch.cat([logits, padding], dim=0)
    
    if temperature <= 0:
        token_id = int(torch.argmax(logits).item())
    else:
        logits = logits / max(1e-6, temperature)
        
        # Check for overflow before softmax
        max_logit = logits.max().item()
        if max_logit > 50:  # Large values can cause overflow
            logits = logits - max_logit + 50
        
        probs = torch.softmax(logits, dim=-1)
        
        # Ensure probabilities are valid
        if not torch.isfinite(probs).all() or probs.sum().item() <= 0:
            print("Warning: Invalid probabilities. Using uniform distribution.")
            probs = torch.ones_like(probs) / probs.shape[0]
        
        probs = _top_k_top_p_filtering(probs, top_k=top_k, top_p=top_p)
        denom = probs.sum()
        if denom <= 0:
            print("Warning: Zero probability mass after filtering. Using original probabilities.")
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = probs / denom
        
        # Safe multinomial sampling
        try:
            token_id = int(torch.multinomial(probs, 1).item())
        except RuntimeError as e:
            print(f"Warning: Multinomial sampling failed: {e}. Using argmax.")
            token_id = int(torch.argmax(probs).item())
    
    # Bounds checking: ensure token_id is within valid vocabulary range
    if token_id < 0 or token_id >= vocab_size:
        print(f"Warning: Generated token ID {token_id} is out of range [0, {vocab_size-1}]. Clamping to valid range.")
        token_id = max(0, min(token_id, vocab_size - 1))
    
    return token_id


def build_prompt_tokens(tokenizer, user_text: str, seq_len: int, device: torch.device) -> torch.Tensor:
    messages = [
        {"role": "user", "content": user_text},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors='pt',
        padding=False,
        truncation=True,
        max_length=seq_len,
        add_generation_prompt=True,
    )
    if isinstance(encoded, dict) or hasattr(encoded, 'input_ids'):
        input_ids = (encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids)
    else:
        input_ids = encoded
    input_ids = input_ids[:, -seq_len:]
    return input_ids.to(device)


def generate(
    model: DeepSeekV3Model,
    tokenizer,
    seq_len: int,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
) -> Tuple[str, list]:
    model.eval()
    eos_id: Optional[int] = tokenizer.eos_token_id
    vocab_size = len(tokenizer)

    # Build initial context
    input_ids = build_prompt_tokens(tokenizer, prompt_text, seq_len, device)

    # Prefill latent cache using the whole prompt
    model.reset_latent_cache()
    model.prefill_with_cache(input_ids)

    generated_ids = input_ids.clone()

    for step_num in range(max_new_tokens):
        try:
            # Next token is generated by stepping with latent cache (recompute K,V from latents)
            last_token_id = generated_ids[0, -1].item()
            step_ids = torch.tensor([[last_token_id]], device=device, dtype=generated_ids.dtype)
            
            # Check for valid token ID before proceeding
            if last_token_id < 0 or last_token_id >= vocab_size:
                print(f"Error: Invalid last token ID {last_token_id}. Breaking generation.")
                break
            
            # Clear any CUDA cache to prevent memory issues
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Try to get logits from model
            try:
                logits = model.step_with_cache(step_ids)
            except RuntimeError as e:
                if "CUDA" in str(e) or "illegal memory access" in str(e):
                    print(f"CUDA error encountered at step {step_num}: {e}")
                    print("Attempting to continue with cache reset...")
                    model.reset_latent_cache()
                    model.prefill_with_cache(generated_ids)
                    logits = model.step_with_cache(step_ids)
                else:
                    raise e
            
            logits_tensor = logits[0]            
            
            # Ensure logits have the correct vocabulary size
            if logits_tensor.shape[0] != vocab_size:
                # Truncate or pad logits to match vocabulary size
                if logits_tensor.shape[0] > vocab_size:
                    logits_tensor = logits_tensor[:vocab_size]
                else:
                    # Pad with very negative values (low probability)
                    padding_size = vocab_size - logits_tensor.shape[0]
                    padding = torch.full((padding_size,), -1e9, device=logits_tensor.device, dtype=logits_tensor.dtype)
                    logits_tensor = torch.cat([logits_tensor, padding], dim=0)
            
            # Clip extreme logits to prevent overflow
            logits_tensor = torch.clamp(logits_tensor, min=-100.0, max=100.0)
            
            next_id = _sample_from_logits(logits_tensor, top_k=top_k, top_p=top_p, temperature=temperature, vocab_size=vocab_size)
            
            next_token = torch.tensor([[next_id]], device=device, dtype=generated_ids.dtype)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if eos_id is not None and next_id == eos_id:
                break

            # Maintain window; trim latent cache consistently by resetting when beyond window
            if generated_ids.shape[1] > seq_len:
                # Rebuild cache from the last seq_len tokens to avoid drift
                generated_ids = generated_ids[:, -seq_len:]
                model.reset_latent_cache()
                model.prefill_with_cache(generated_ids)
                
        except Exception as e:
            print(f"Error at generation step {step_num}: {e}")
            print("Stopping generation due to error.")
            break

    # Decode continuation beyond initial context
    init_len = input_ids.shape[1]
    full_sequence = generated_ids[0].tolist()
    cont_tokens = full_sequence[init_len:]
    
    # Additional safety check before decoding
    valid_cont_tokens = []
    for token_id in cont_tokens:
        if 0 <= token_id < vocab_size:
            valid_cont_tokens.append(token_id)
        else:
            print(f"Warning: Skipping invalid token ID {token_id} during decoding (vocab_size: {vocab_size})")
    
    text = tokenizer.decode(valid_cont_tokens, skip_special_tokens=True)
    return text, full_sequence


def main():
    cli = parse_args()

    device = torch.device(cli.device)
    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(cli.tokenizer_path, trust_remote_code = True)
    except Exception as e:
        print(f"Error loading tokenizer from {cli.tokenizer_path}: {e}")
        return
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    args_for_model = build_args_from_tokenizer(cli, tokenizer)

    # Debug information
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Model vocabulary size: {args_for_model.vocab_size}")
    print(f"Tokenizer EOS token ID: {tokenizer.eos_token_id}")
    print(f"Tokenizer PAD token ID: {tokenizer.pad_token_id}")
    
    # Validate vocab size match
    if len(tokenizer) != args_for_model.vocab_size:
        print(f"WARNING: Vocabulary size mismatch! Tokenizer: {len(tokenizer)}, Model: {args_for_model.vocab_size}")
        print("This could cause out-of-range token generation errors.")

    try:
        model = load_model(cli.model_save_path, args_for_model, device)
    except Exception as e:
        print(f"Error loading model from {cli.model_save_path}: {e}")
        return

    # Validate model head dimensions
    print(f"Model head shape: {model.head.shape}")
    expected_head_shape = (args_for_model.vocab_size, args_for_model.d_model)
    if model.head.shape != expected_head_shape:
        print(f"WARNING: Model head shape {model.head.shape} doesn't match expected {expected_head_shape}")

    print(f"\nGenerating text for prompt: '{cli.prompt}'")
    print("=" * 50)
    
    try:
        if cli.use_specdec and args_for_model.mtp_depth > 0 and getattr(model, 'mtp', None) is not None:
            text, _ = generate_speculative(
                model=model,
                tokenizer=tokenizer,
                seq_len=args_for_model.seq_len,
                prompt_text=cli.prompt,
                device=device,
                max_new_tokens=cli.max_new_tokens,
                top_k=cli.top_k,
                top_p=cli.top_p,
                temperature=cli.temperature,
                mtp_depth=args_for_model.mtp_depth,
            )
        else:
            text, _ = generate(
                model=model,
                tokenizer=tokenizer,
                seq_len=args_for_model.seq_len,
                prompt_text=cli.prompt,
                device=device,
                max_new_tokens=cli.max_new_tokens,
                top_k=cli.top_k,
                top_p=cli.top_p,
                temperature=cli.temperature,
            )
        print(text)
    except Exception as e:
        print(f"Error during text generation: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Reduce --max_new_tokens to a smaller number (e.g., 10)")
        print("2. Check if the model checkpoint is compatible with this code")
        print("3. Verify the tokenizer path is correct")
        print("4. Check that model weights are not corrupted")
        return


if __name__ == '__main__':
    main()
