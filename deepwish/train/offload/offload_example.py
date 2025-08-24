import torch
import torch.nn as nn
import torch.distributed as dist
import os
import sys
import time
import argparse

def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU mode
        return 0, 1, 0

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_gpu_memory_usage():
    """Returns the currently used GPU memory in GiB."""
    # Report memory for the current CUDA device
    current = torch.cuda.current_device()
    return torch.cuda.memory_allocated(current) / (1024**3)

class BigModel(nn.Module):
    """A large model with multiple layers to test checkpointing."""
    def __init__(self, hidden_dim, num_layers, bottleneck_dim=None):
        super().__init__()
        # Use a smaller bottleneck to balance params vs activations
        if bottleneck_dim is None:
            bottleneck_dim = hidden_dim // 2
        self.bottleneck_dim = bottleneck_dim
        # Each layer: down-project then up-project
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.ReLU()
            )
            for _ in range(num_layers)
        ])
        # Add a simple head for final output
        self.head = nn.Parameter(torch.randn(hidden_dim, hidden_dim) / (hidden_dim ** 0.5))

    def forward(self, x):
        # This is a standard forward pass. The custom training loop will handle checkpointing.
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            print(f"[Vanilla] After layer {idx}, GPU memory: {get_gpu_memory_usage():.2f} GiB")
        return x @ self.head

def run_distributed_offload_step(model, input_tensor, target_tensor, args, adam_states, step):
    """
    Implements the exact same distributed offloading procedure as train_fa2_cce_offload.py
    """
    device = next(model.parameters()).device
    betas = (args.adam_beta1, args.adam_beta2)
    eps = args.adam_eps
    
    # Get distributed training info
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    # local print only on rank 0
    def _print(*pmsg, **pkwargs):
        if rank == 0:
            print(*pmsg, **pkwargs)

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
    
    modules = list(model.layers)
    offloaded = []
    transfer_stream = torch.cuda.Stream()
    
    x = input_tensor
    _print(f"[Distributed] Initial GPU memory: {get_gpu_memory_usage():.2f} GiB")

    # Forward pass with activation offloading
    with torch.no_grad():
        for m in modules:
            with torch.cuda.stream(transfer_stream):
                buf = x.to('cpu', non_blocking=True).pin_memory()                    
            ev_act = torch.cuda.Event()
            ev_act.record(transfer_stream)
            offloaded.append((buf, ev_act, m))
            x = m(x)
            _print(f"[Distributed] After forward layer, GPU memory: {get_gpu_memory_usage():.2f} GiB")

    torch.cuda.empty_cache()
    
    hidden = x.clone().detach().requires_grad_(True)

    # Loss computation
    output = hidden @ model.head
    loss_main = nn.MSELoss()(output, target_tensor)
    _print(f"[Distributed] After loss calc, GPU memory: {get_gpu_memory_usage():.2f} GiB")
    
    total_loss = loss_main
    
    # Backward pass for head
    total_loss.backward()

    # Reduce-scatter and update head parameters
    with torch.no_grad():
        head_params = [model.head]
        
        # Reduce-scatter gradients
        _reduce_scatter_gradients(head_params)
        
        # Update only this rank's shard
        for p in head_params:
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
        _all_gather_params(head_params)

    grad = hidden.grad.clone()
    del hidden, total_loss, loss_main
    torch.cuda.empty_cache()

    # Layer-wise backward with distributed updates
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
        
        # Prefetch optimizer states for this rank's parameter shards
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
        _print(f"[Distributed] After layer backward cleanup, GPU memory: {get_gpu_memory_usage():.2f} GiB")

    final_loss = nn.MSELoss()(x.detach() @ model.head, target_tensor)
    return final_loss.item()

def run_vanilla_training_step(model, input_tensor, target_tensor, optimizer, loss_fn):
    """Performs a standard PyTorch training step."""
    print(f"[Vanilla] Before zero_grad: {get_gpu_memory_usage():.2f} GiB")
    optimizer.zero_grad()
    print(f"[Vanilla] After zero_grad: {get_gpu_memory_usage():.2f} GiB")
    output = model(input_tensor)
    print(f"[Vanilla] After forward: {get_gpu_memory_usage():.2f} GiB")
    loss = loss_fn(output, target_tensor)
    print(f"[Vanilla] After loss calc: {get_gpu_memory_usage():.2f} GiB")
    loss.backward()
    print(f"[Vanilla] After backward: {get_gpu_memory_usage():.2f} GiB")
    optimizer.step()
    print(f"[Vanilla] After optimizer.step: {get_gpu_memory_usage():.2f} GiB")

def parse_args():
    parser = argparse.ArgumentParser(description="Test distributed offloading example")
    parser.add_argument('--hidden_dim', type=int, default=10240, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=10, help='Number of layers')
    parser.add_argument('--batch_size', type=int, default=10240, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--skip_vanilla', action='store_true', help='Skip vanilla benchmark')
    return parser.parse_args()

def main():
    # Initialize distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    try:
        args = parse_args()
        
        # Only print from rank 0
        def print_rank0(*msg):
            if rank == 0:
                print(*msg)

        if not torch.cuda.is_available():
            print_rank0("Skipping test: CUDA device not available.")
            return

        # Configuration
        HIDDEN_DIM = args.hidden_dim
        BATCH_SIZE = args.batch_size
        NUM_LAYERS = args.num_layers
        BOTTLENECK_DIM = HIDDEN_DIM // 2
        DEVICE = device

        print_rank0("--- Test Configuration ---")
        print_rank0(f"World Size: {world_size}, Rank: {rank}, Local Rank: {local_rank}")
        print_rank0(f"Device: {torch.cuda.get_device_name(DEVICE)}")
        print_rank0(f"Hidden Dim: {HIDDEN_DIM}, Bottleneck Dim: {BOTTLENECK_DIM}, Batch Size: {BATCH_SIZE}, Layers: {NUM_LAYERS}\n")
        
        # Expected memory calculations
        exp_weight_mem = NUM_LAYERS * (HIDDEN_DIM * BOTTLENECK_DIM * 2) * 4 / (1024**3)
        exp_act_mem = BATCH_SIZE * HIDDEN_DIM * 4 / (1024**3)
        exp_vanilla_mem = exp_weight_mem + exp_act_mem * NUM_LAYERS
        exp_custom_mem = exp_weight_mem + exp_act_mem
        print_rank0(f"Expected weights memory: {exp_weight_mem:.2f} GiB")
        print_rank0(f"Expected activation memory per layer: {exp_act_mem:.2f} GiB")
        print_rank0(f"Expected vanilla peak memory: {exp_vanilla_mem:.2f} GiB")
        print_rank0(f"Expected distributed peak memory: {exp_custom_mem:.2f} GiB")

        # Create test data
        input_tensor = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=DEVICE)
        target_tensor = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=DEVICE)
        loss_fn = nn.MSELoss()

        # --- 1. Vanilla PyTorch Benchmark (only on rank 0 to avoid OOM) ---
        if not args.skip_vanilla and rank == 0:
            print_rank0("--- Running Vanilla PyTorch Benchmark ---")
            model_vanilla = BigModel(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
            optimizer_vanilla = torch.optim.Adam(model_vanilla.parameters(), lr=args.lr)

            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(DEVICE)
                start_time = time.time()

                run_vanilla_training_step(model_vanilla, input_tensor, target_tensor, optimizer_vanilla, loss_fn)
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                peak_mem_vanilla = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
                time_vanilla = end_time - start_time
                print_rank0("✅ Vanilla training step completed.")
                print_rank0(f"⏱️ Time Taken: {time_vanilla:.4f} seconds")
                print_rank0(f"📈 Peak GPU Memory: {peak_mem_vanilla:.2f} GiB")

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print_rank0(f"❌ FAILED: Vanilla training failed with error: {e}")
            finally:
                del model_vanilla, optimizer_vanilla
                torch.cuda.empty_cache()

            print_rank0("\n" + "="*40 + "\n")

        # --- 2. Distributed Offloading Benchmark ---
        print_rank0("--- Running Distributed Offloading Benchmark ---")
        model_distributed = BigModel(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
        
        # We manually shard parameters, no DDP wrapper needed
        real_model = model_distributed
        
        # Store optimizer state on CPU pinned memory for async transfer
        adam_states = {p: {'m': torch.zeros_like(p, device='cpu').pin_memory(),
                          'v': torch.zeros_like(p, device='cpu').pin_memory()} for p in real_model.parameters()}
        
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(DEVICE)
            start_time = time.time()

            step = 1
            loss_val = run_distributed_offload_step(real_model, input_tensor, target_tensor, args, adam_states, step)
            
            torch.cuda.synchronize()
            end_time = time.time()

            peak_mem_custom = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
            time_custom = end_time - start_time
            print_rank0(f"\n✅ Distributed training step completed with loss: {loss_val:.6f}")
            print_rank0(f"⏱️ Time Taken: {time_custom:.4f} seconds")
            print_rank0(f"📈 Peak GPU Memory: {peak_mem_custom:.2f} GiB")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print_rank0(f"❌ FAILED: Distributed training failed with error: {e}")
        finally:
            del model_distributed, input_tensor, target_tensor, loss_fn
            torch.cuda.empty_cache()
            print_rank0(f"\nFinal GPU Memory after cleanup: {get_gpu_memory_usage():.2f} GiB")

    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
