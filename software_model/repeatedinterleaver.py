import torch
import time
import statistics
import numpy as np
from utils import size
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from hardware_model.device import Device


@torch.compile
def repeat_interleave_gpu(input: torch.Tensor, repeats: int, dim: int):
    return torch.repeat_interleave(input, repeats=repeats, dim=dim)

# def repeat_kv(x, n_rep):

#     batch_size, seq_len, n_kv_heads, head_dim = x.shape
#     if n_rep == 1:
#         return x
#     else:
#         # (m, seq_len, n_kv_heads, 1, head_dim)
#         # --> (m, seq_len, n_kv_heads, n_rep, head_dim)
#         # --> (m, seq_len, n_kv_heads * n_rep, head_dim)
#         return (
#             x[:, :, :, None, :]
#             .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
#             .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
#         )

class RepeatedInterleave(Operator):
    def __init__(self, data_type: DataType, repeats: int, dim: int):
        super().__init__(0, 0, 0, 0, data_type)
        self.repeats = repeats
        self.dim = dim
        self.shape = None

    def __call__(self, input: Tensor) -> Tensor:
        assert self.data_type == input.data_type
        self.shape = list(input.shape)
        self.shape[self.dim] *= self.repeats
        self.input_shape = input.shape
        self.M = size(input.shape)
        self.N = size(self.shape)
        return Tensor(self.shape, self.data_type)

    def roofline_model(self, pcb_module: Device):
        # Just read + write, no FLOPs
        self.io_count = (self.M + self.N) * self.data_type.word_size
        self.flop_count = 0
        self.roofline_latency = self.io_count / min(
            pcb_module.io_module.bandwidth,
            pcb_module.compute_module.l2_bandwidth_per_cycle * pcb_module.compute_module.clock_freq,
        )
        return self.roofline_latency

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str = "heuristic-GPU"):
        if compile_mode != "heuristic-GPU":
            raise ValueError("RepeatedInterleave only supports heuristic-GPU mode.")
        
        # Simulate basic memory movement: read + write
        read_bytes = self.M * self.data_type.word_size
        write_bytes = self.N * self.data_type.word_size
        total_bytes = read_bytes + write_bytes

        total_cycles = total_bytes / (
            pcb_module.io_module.bandwidth / pcb_module.compute_module.clock_freq
        )

        self.best_latency = total_cycles / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        return self.latency

    def run_on_gpu(self):
        assert self.shape is not None
        input = torch.randn(self.input_shape, dtype=torch.float16, device="cuda")
        latencies = []

        for _ in range(3):
            _ = repeat_interleave_gpu(input, self.repeats, self.dim)
            torch.cuda.synchronize()

        for _ in range(self.iterations):
            start = time.time()
            output = repeat_interleave_gpu(input, self.repeats, self.dim)
            torch.cuda.synchronize()
            end = time.time()
            assert output.shape[self.dim] == input.shape[self.dim] * self.repeats
            latencies.append(end - start)

        self.latency_on_gpu = statistics.median(latencies)
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        latencies = []
        a = torch.randn(1, 1, 1, device="cuda")
        for _ in range(50):
            start = time.time()
            _ = repeat_interleave_gpu(a, 4, 1)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        print(latencies)
        return avg_overhead
