import torch
import torch.distributed as dist
import sys
import os

# Add project root to Python path for all imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.model import triton_cce_loss


def run_offload_deepseek_step(model, inp_ids, target_main, tgt_matrix, args, adam_states, step):
    """
    Implements distributed parameter sharding with activation offloading
    following the exact same pattern as offload_example.py
    """
    device = next(model.parameters()).device
    betas = (args.adam_beta1, args.adam_beta2)
    eps = args.adam_eps
    
    # Get distributed training info
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    def _get_param_shard(param, rank, world_size):
        """Get the shard of parameters this rank should handle."""
        total_elements = param.numel()
        elements_per_rank = total_elements // world_size
        start_idx = rank * elements_per_rank
        end_idx = start_idx + elements_per_rank if rank < world_size - 1 else total_elements
        return start_idx, end_idx

    def _apply_adam_update_shard(param_shard, m_shard, v_shard, grad_shard, t, lr, betas, eps, weight_decay):
        """Applies Adam update to a parameter shard."""
        beta1, beta2 = betas
        
        m_shard.mul_(beta1).add_(grad_shard, alpha=1 - beta1)
        v_shard.mul_(beta2).addcmul_(grad_shard, grad_shard, value=1 - beta2)
        
        m_hat = m_shard / (1 - beta1 ** t)
        v_hat = v_shard / (1 - beta2 ** t)
        
        if weight_decay > 0:
            param_shard.add_(param_shard, alpha=-lr * weight_decay)

        param_shard.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        return m_shard, v_shard

    def _reduce_scatter_gradients(params):
        """Reduce-scatter gradients across GPUs."""
        for p in params:
            if p.grad is None:
                continue
            
            # Flatten gradient
            grad_flat = p.grad.data.view(-1)
            total_elements = grad_flat.numel()
            
            if world_size > 1:
                # Create output tensor for this rank's shard
                elements_per_rank = total_elements // world_size
                start_idx = rank * elements_per_rank
                end_idx = start_idx + elements_per_rank if rank < world_size - 1 else total_elements
                shard_size = end_idx - start_idx
                
                # Prepare input list for reduce_scatter
                input_list = [grad_flat[i * elements_per_rank:(i + 1) * elements_per_rank] 
                             for i in range(world_size - 1)]
                # Handle last shard which might be larger
                input_list.append(grad_flat[(world_size - 1) * elements_per_rank:])
                
                # Output tensor for this rank's portion
                output = torch.zeros(shard_size, device=device, dtype=grad_flat.dtype)
                
                # Reduce-scatter operation
                dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)
                
                # Store the reduced shard back
                p.grad_shard = output
            else:
                p.grad_shard = grad_flat

    def _all_gather_params(params):
        """All-gather parameters after updates."""
        for p in params:
            if world_size > 1:
                # Get the parameter shards from all ranks
                param_flat = p.data.view(-1)
                total_elements = param_flat.numel()
                elements_per_rank = total_elements // world_size
                
                # Each rank contributes its shard
                start_idx = rank * elements_per_rank
                end_idx = start_idx + elements_per_rank if rank < world_size - 1 else total_elements
                my_shard = param_flat[start_idx:end_idx]
                
                # Create list to gather into - each element should be the size of a shard
                gather_list = []
                for i in range(world_size):
                    if i < world_size - 1:
                        shard_size = elements_per_rank
                    else:
                        shard_size = total_elements - i * elements_per_rank
                    gather_list.append(torch.zeros(shard_size, device=device, dtype=param_flat.dtype))
                
                # All-gather operation
                dist.all_gather(gather_list, my_shard)
                
                # Reconstruct full parameter
                full_param = torch.cat(gather_list)
                p.data.copy_(full_param.view(p.data.shape))
    
    modules = [model.token_embed] + list(model.layers)
    offloaded = []
    transfer_stream = torch.cuda.Stream()
    
    x = inp_ids

    # Forward pass with activation offloading (same pattern as offload_example.py)
    with torch.no_grad():
        for m in modules:
            with torch.cuda.stream(transfer_stream):
                buf = x.to('cpu', non_blocking=True).pin_memory()                    
            ev_act = torch.cuda.Event()
            ev_act.record(transfer_stream)
            offloaded.append((buf, ev_act, m))
            x = m(x)

    torch.cuda.empty_cache()
    
    hidden = x.clone().detach().requires_grad_(True)

    # Loss computation
    loss_main = triton_cce_loss(hidden, model.head, target_main.reshape(-1), ignore_index=model.token_embed.padding_idx)

    loss_mtp = torch.tensor(0.0, device=device)
    if model.mtp:
        emb_mtp = model.mtp(hidden)
        loss_mtp_val = 0.0 * model.head.sum()
        for j in range(model.mtp.depth):
            tgt_j = tgt_matrix[:, :, j].reshape(-1)
            loss_mtp_val += triton_cce_loss(emb_mtp[:, :, j, :], model.head, tgt_j, ignore_index=model.token_embed.padding_idx)
        loss_mtp = loss_mtp_val / model.mtp.depth
    
    total_loss = loss_main + args.mtp_weight * loss_mtp
    
    # Backward pass for head/MTP
    total_loss.backward()

    # Reduce-scatter and update head/MTP parameters
    with torch.no_grad():
        head_and_mtp_params = list(model.mtp.parameters()) + [model.head] if model.mtp else [model.head]
        
        # Reduce-scatter gradients
        _reduce_scatter_gradients(head_and_mtp_params)
        
        # Update only this rank's shard
        for p in head_and_mtp_params:
            if not hasattr(p, 'grad_shard'):
                continue
                
            state = adam_states[p]
            start_idx, end_idx = _get_param_shard(p, rank, world_size)
            
            # Get CPU optimizer state for this shard
            m_cpu_shard = state['m'].view(-1)[start_idx:end_idx]
            v_cpu_shard = state['v'].view(-1)[start_idx:end_idx]
            
            # Move to GPU
            m_gpu_shard = m_cpu_shard.to(device)
            v_gpu_shard = v_cpu_shard.to(device)
            
            # Get parameter shard
            param_shard = p.data.view(-1)[start_idx:end_idx]
            
            # Apply Adam update to shard
            m_gpu_shard, v_gpu_shard = _apply_adam_update_shard(
                param_shard, m_gpu_shard, v_gpu_shard, p.grad_shard, 
                step, args.lr, betas, eps, args.weight_decay
            )
            
            # Move optimizer state back to CPU
            state['m'].view(-1)[start_idx:end_idx].copy_(m_gpu_shard.to('cpu'))
            state['v'].view(-1)[start_idx:end_idx].copy_(v_gpu_shard.to('cpu'))
            
            p.grad = None
            delattr(p, 'grad_shard')
        
        # All-gather updated parameters
        _all_gather_params(head_and_mtp_params)

    grad = hidden.grad.clone()
    del hidden, total_loss, loss_main, loss_mtp
    torch.cuda.empty_cache()

    # Layer-wise backward with distributed updates (same pattern as offload_example.py)
    for buf, ev_act, m in reversed(offloaded):
        torch.cuda.current_stream().wait_event(ev_act)
        inp = buf.to(device, non_blocking=True)
        if not torch.is_floating_point(inp):
            inp.requires_grad = False
        else:
            inp.requires_grad = True

        with torch.enable_grad():
            out = m(inp)
        
        torch.autograd.backward(out, grad_tensors=[grad])
        
        # Prefetch optimizer states for this rank's parameter shards (during backward like offload_example.py)
        for p in m.parameters():
            start_idx, end_idx = _get_param_shard(p, rank, world_size)
            cpu_state = adam_states[p]
            # Only transfer the shard this rank needs
            adam_states[p]['_m_gpu_shard'] = cpu_state['m'].view(-1)[start_idx:end_idx].to(device, non_blocking=True)
            adam_states[p]['_v_gpu_shard'] = cpu_state['v'].view(-1)[start_idx:end_idx].to(device, non_blocking=True)
            adam_states[p]['_shard_info'] = (start_idx, end_idx)
        
        # Reduce-scatter gradients for this layer
        layer_params = list(m.parameters())
        _reduce_scatter_gradients(layer_params)
        
        with torch.no_grad():
            for p in layer_params:
                if not hasattr(p, 'grad_shard'):
                    continue
                    
                state = adam_states[p]
                start_idx, end_idx = state['_shard_info']
                
                # Use prefetched optimizer state shards
                m_gpu_shard = state.pop('_m_gpu_shard')
                v_gpu_shard = state.pop('_v_gpu_shard')
                
                # Get parameter shard
                param_shard = p.data.view(-1)[start_idx:end_idx]
                
                # Apply Adam update
                m_gpu_shard, v_gpu_shard = _apply_adam_update_shard(
                    param_shard, m_gpu_shard, v_gpu_shard, p.grad_shard,
                    step, args.lr, betas, eps, args.weight_decay
                )
                
                # Asynchronously move optimizer state back to CPU
                with torch.cuda.stream(transfer_stream):
                    state['m'].view(-1)[start_idx:end_idx].copy_(m_gpu_shard.to('cpu', non_blocking=True))
                    state['v'].view(-1)[start_idx:end_idx].copy_(v_gpu_shard.to('cpu', non_blocking=True))
                
                delattr(p, 'grad_shard')
                state.pop('_shard_info')
        
            # All-gather updated parameters for this layer
            _all_gather_params(layer_params)
            
            m.zero_grad(set_to_none=True)

        if inp.grad is not None:
            grad = inp.grad.clone()

        del inp, out, buf, ev_act
        torch.cuda.empty_cache()

    final_loss_main = triton_cce_loss(x.detach(), model.head, target_main.reshape(-1), ignore_index=model.token_embed.padding_idx)
    return final_loss_main.item(), 0.0