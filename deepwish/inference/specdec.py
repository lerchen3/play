import torch
from typing import Tuple, Optional


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
    if not torch.isfinite(logits).all():
        logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
    if logits.shape[0] != vocab_size:
        if logits.shape[0] > vocab_size:
            logits = logits[:vocab_size]
        else:
            padding = torch.full((vocab_size - logits.shape[0],), -1e9, device=logits.device, dtype=logits.dtype)
            logits = torch.cat([logits, padding], dim=0)
    if temperature <= 0:
        token_id = int(torch.argmax(logits).item())
    else:
        logits = logits / max(1e-6, temperature)
        max_logit = logits.max().item()
        if max_logit > 50:
            logits = logits - max_logit + 50
        probs = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all() or probs.sum().item() <= 0:
            probs = torch.ones_like(probs) / probs.shape[0]
        probs = _top_k_top_p_filtering(probs, top_k=top_k, top_p=top_p)
        denom = probs.sum()
        if denom <= 0:
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = probs / denom
        try:
            token_id = int(torch.multinomial(probs, 1).item())
        except RuntimeError:
            token_id = int(torch.argmax(probs).item())
    if token_id < 0 or token_id >= vocab_size:
        token_id = max(0, min(token_id, vocab_size - 1))
    return token_id


@torch.no_grad()
def generate_speculative(
    model,
    tokenizer,
    seq_len: int,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    mtp_depth: int,
) -> Tuple[str, list]:
    model.eval()
    eos_id: Optional[int] = tokenizer.eos_token_id
    vocab_size = len(tokenizer)

    # Build initial context
    messages = [
        {"role": "user", "content": prompt_text},
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
    input_ids = input_ids.to(device)

    # Prefill caches for base model and capture prompt hidden
    model.reset_latent_cache()
    h_prompt = model.prefill_with_cache(input_ids)

    # Prefill MTP caches if available
    use_mtp = (getattr(model, 'mtp', None) is not None) and (mtp_depth and mtp_depth > 0)
    if use_mtp:
        model.mtp.reset_latent_cache()
        model.mtp.prefill_with_cache(h_prompt)

    generated_ids = input_ids.clone()

    steps = 0
    while steps < max_new_tokens:
        last_token_id = generated_ids[0, -1].item()
        step_ids = torch.tensor([[last_token_id]], device=device, dtype=generated_ids.dtype)

        # Base step: get logits for next token t1 from current state
        logits_next, h_step = model.step_with_cache(
            step_ids,
            return_all_logits=False,
            return_hidden=True,
            update_cache=True,
        )
        # Sample and commit t1
        token_id_1 = _sample_from_logits(logits_next[0], top_k=top_k, top_p=top_p, temperature=temperature, vocab_size=vocab_size)
        tok_tensor = torch.tensor([[token_id_1]], device=device, dtype=generated_ids.dtype)
        generated_ids = torch.cat([generated_ids, tok_tensor], dim=1)
        steps += 1
        if eos_id is not None and token_id_1 == eos_id:
            break

        # Maintain window; if overflow, rebuild caches
        if generated_ids.shape[1] > seq_len:
            generated_ids = generated_ids[:, -seq_len:]
            model.reset_latent_cache()
            h_prompt = model.prefill_with_cache(generated_ids)
            if use_mtp:
                model.mtp.reset_latent_cache()
                model.mtp.prefill_with_cache(h_prompt)
            continue

        # Advance base cache with the committed t1 and get logits for t2
        base_logits_for_next, h_after_commit = model.step_with_cache(
            tok_tensor,
            return_all_logits=False,
            return_hidden=True,
            update_cache=True,
        )

        # If no MTP, continue to next step
        if not use_mtp:
            continue

        # Keep MTP cache in sync with committed token
        model.mtp.step_with_cache_hidden(h_after_commit, update_cache=True, return_hidden=False)

        # Propose up to mtp_depth tokens using MTP block starting from committed hidden
        proposals = []
        mtp_hidden_in = h_after_commit
        for _ in range(mtp_depth):
            h_mtp = model.mtp.step_with_cache_hidden(mtp_hidden_in, update_cache=False, return_hidden=True)
            logits_prop = torch.matmul(h_mtp[:, -1, :], model.head.t())  # (B, V)
            prop_token = _sample_from_logits(logits_prop[0], top_k=top_k, top_p=top_p, temperature=temperature, vocab_size=vocab_size)
            proposals.append(prop_token)
            # Feed last hidden to next MTP depth
            mtp_hidden_in = h_mtp[:, -1:, :]

        # Verify proposals sequentially (these correspond to t2..tK)
        i = 0
        while i < len(proposals) and steps < max_new_tokens:
            proposed_tok = proposals[i]
            # Draw base sample for this position
            base_sample = _sample_from_logits(base_logits_for_next[0], top_k=top_k, top_p=top_p, temperature=temperature, vocab_size=vocab_size)
            if base_sample == proposed_tok:
                # Accept and commit proposed token
                tok_acc = torch.tensor([[proposed_tok]], device=device, dtype=generated_ids.dtype)
                generated_ids = torch.cat([generated_ids, tok_acc], dim=1)
                steps += 1
                if eos_id is not None and proposed_tok == eos_id:
                    i += 1
                    break
                # Advance base cache and get logits for next verification
                logits_after_commit, h_after_commit = model.step_with_cache(
                    tok_acc,
                    return_all_logits=False,
                    return_hidden=True,
                    update_cache=True,
                )
                base_logits_for_next = logits_after_commit

                # Advance MTP cache with the committed hidden
                model.mtp.step_with_cache_hidden(h_after_commit, update_cache=True, return_hidden=False)
                # Maintain window
                if generated_ids.shape[1] > seq_len:
                    generated_ids = generated_ids[:, -seq_len:]
                    model.reset_latent_cache()
                    h_prompt = model.prefill_with_cache(generated_ids)
                    if use_mtp:
                        model.mtp.reset_latent_cache()
                        model.mtp.prefill_with_cache(h_prompt)
                    break
                i += 1
            else:
                # Reject remainder; commit correction token sampled from base
                corr_tensor = torch.tensor([[base_sample]], device=device, dtype=generated_ids.dtype)
                generated_ids = torch.cat([generated_ids, corr_tensor], dim=1)
                steps += 1
                # Advance base cache to reflect correction
                _, h_after_corr = model.step_with_cache(
                    corr_tensor,
                    return_all_logits=False,
                    return_hidden=True,
                    update_cache=True,
                )
                # Advance MTP cache to stay aligned
                model.mtp.step_with_cache_hidden(h_after_corr, update_cache=True, return_hidden=False)
                if eos_id is not None and base_sample == eos_id:
                    break
                if generated_ids.shape[1] > seq_len:
                    generated_ids = generated_ids[:, -seq_len:]
                    model.reset_latent_cache()
                    h_prompt = model.prefill_with_cache(generated_ids)
                    if use_mtp:
                        model.mtp.reset_latent_cache()
                        model.mtp.prefill_with_cache(h_prompt)
                break  # end verification group on first mismatch

    # Decode continuation beyond initial context
    init_len = input_ids.shape[1]
    full_sequence = generated_ids[0].tolist()
    cont_tokens = full_sequence[init_len:]
    valid_cont_tokens = [t for t in cont_tokens if 0 <= t < vocab_size]
    text = tokenizer.decode(valid_cont_tokens, skip_special_tokens=True)
    return text, full_sequence