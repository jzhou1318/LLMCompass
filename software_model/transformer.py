from software_model.operators import (
    Operator,
    Reshape,
    Concat,
    Transpose,
)
from software_model.matmul import Matmul, BatchedMatmul
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU


from software_model.utils import Tensor, DataType
from software_model.communication_primitives import AllReduceMultiPCB
from math import ceil
from typing import List
from hardware_model.system import System


class TransformerBlockInitComputationTP(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count
        # parameters per device
        d = d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)
        # operators per device
        # # multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        # # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def __call__(self, X: Tensor) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, s, d = X.shape
        assert d == self.d_model
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h

        # multi-head attention
        Q = self.Q_proj(X, self.Wq)  # [b, s, d / dev_cnt]
        assert Q.shape == [b, s, d // dev_cnt]
        K = self.K_proj(X, self.Wk)  # [b, s, d / dev_cnt]
        V = self.V_proj(X, self.Wv)  # [b, s, d / dev_cnt]
        Q = self.Q_reshape(Q, [b, s, h // dev_cnt, d_h])
        K = self.K_reshape(K, [b, s, h // dev_cnt, d_h])
        V = self.V_reshape(V, [b, s, h // dev_cnt, d_h])
        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h]
        assert Q_T.shape == [b, h // dev_cnt, s, d_h]
        K_T = self.K_transpose(K, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, s]
        assert K_T.shape == [b, h // dev_cnt, d_h, s]
        V_T = self.V_transpose(V, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h]
        assert V_T.shape == [b, h // dev_cnt, s, d_h]
        A = self.Q_mul_K(Q_T, K_T)  # [b, h / dev_cnt, s, s]
        assert A.shape == [b, h // dev_cnt, s, s]
        A_prob = self.A_softmax(A)
        H = self.A_mul_V(A_prob, V_T)  #  [b, h / dev_cnt, s, d_h]
        assert H.shape == [b, h // dev_cnt, s, d_h]
        H = self.H_transpose(H, [0, 2, 1, 3])  #  [b, s, h / dev_cnt, d_h]
        assert H.shape == [b, s, h // dev_cnt, d_h]
        H = self.H_reshape(H, [b, s, d // dev_cnt])
        assert H.shape == [b, s, d // dev_cnt]
        H0 = self.H_matmul0(H, self.W0)  #  [b, s, d]
        assert H0.shape == [b, s, d]
        H0 = self.layer_norm0(H0)
        assert H0.shape == [b, s, d]
        if dev_cnt > 1:
            H0 = self.allreduce_mha(H0)

        # feed-forward network
        H1 = self.H_matmul1(H0, self.W1)  # [b, s, 4 * d / dev_cnt]
        assert H1.shape == [b, s, 4 * d // dev_cnt]
        H1 = self.H_gelu(H1)
        H2 = self.H_matmul2(H1, self.W2)  #  [b, s, d]
        assert H2.shape == [b, s, d]
        H2 = self.layer_norm1(H2)
        if dev_cnt > 1:
            H2 = self.allreduce_ffn(H2)

        assert H2.shape == [b, s, d]
        return H2

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        qkv_latency = 3 * (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_total_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        )
        self.roofline_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.roofline_latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        return self.roofline_latency

    def compile_and_simulate(self, system: System, compile_mode: str):
        device = system.device
        interconnect = system.interconnect

        # matmul
        print("simulating qkv")
        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        print("simulating q_mul_k")
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )

        softmax_latency = (
            self.A_softmax.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.softmax
        )

        print("simulating a_mul_v")
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        print("simulating h_matmul0")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        layernorm0_latency = (
            self.layer_norm0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )

        print("simulating h1_matmul1")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )

        # gelu
        gelu_latency = (
            self.H_gelu.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.gelu
        )

        print("simulating h2_matmul2")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )

        layernorm1_latency = (
            self.layer_norm1.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )

        print("finish matmul simulation")

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        normlization_total_latency = softmax_latency + layernorm0_latency + layernorm1_latency

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        # print("breakdown:")
        # print(
        #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        # )
        # print("total:")
        # print(
        #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        # )
        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        self.simluate_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.latency

    def run_on_gpu(self):
        # matmul
        qkv_latency = (
            self.Q_proj.run_on_gpu()  # - self.Q_proj.gpu_kernel_launch_overhead()
        ) * 3
        q_mul_k_latency = (
            self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
        )
        a_mul_v_latency = (
            self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
        )
        h_matmul0_latency = (
            self.H_matmul0.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
        )
        h1_matmul1_latency = (
            self.H_matmul1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
        )
        h2_matmul2_latency = (
            self.H_matmul2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
        )
        layernorm_latency = (
            self.layer_norm0.run_on_gpu()
            - self.layer_norm0.gpu_kernel_launch_overhead()
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.run_on_gpu()  # - self.H_gelu.gpu_kernel_launch_overhead()
        )

        # allreduce
        allreduce_total_latency = 0

        # others

        # print
        print("breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.latency_on_gpu = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        return self.latency_on_gpu


class TransformerBlockAutoRegressionTP(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count
        # parameters per device
        d = d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)
        # operators per device
        # # multi-head attention
        self.Q_proj = Matmul(data_type, 
                                addresses = [int(0x14ab43e06400), int(0x14ab43e0c800), int(0x14ac8e000000)],
                                function_name = "Q_proj")
        self.K_proj = Matmul(data_type, 
                                addresses = [int(0x14ab43e06400), int(0x14ab43e0e100), int(0x14ac8e001900)],
                                function_name = "K_proj")
        self.V_proj = Matmul(data_type, 
                                addresses = [int(0x14ab43e06400), int(0x14ab43e0fa00), int(0x14ac8e003200)],
                                function_name = "V_proj")
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)
        self.Q_mul_K = BatchedMatmul(data_type,
                                        addresses = [int(0x14ab43e0c800), int(0x14a9a09fac00), int(0x14ab43e0e100)],
                                        function_name = "Q_mul_K")
        self.A_softmax = Softmax(data_type,
                                        addresses = [int(0x14a9a09fb400), 0, 0],
                                        function_name = "softmax")
        self.A_mul_V = BatchedMatmul(data_type,
                                        addresses = [int(0x14a9a09fb400), int(0x14ab43e1f400), int(0x14ab43e0fa00)],
                                        function_name = "A_mul_V")
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type,
                                        addresses = [int(0x14ab43e25800), int(0x14ab43e1f400), int(0x14ac90640000)],
                                        function_name = "H_matmul0")
        self.layer_norm0 = LayerNorm(data_type,
                                        addresses = [int(0x14ab43e1f400), 0, 0],
                                        function_name = "layer_norm0")
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        # # feed-forward network
        self.H_matmul1 = Matmul(data_type, 
                                addresses = [int(0x14874b2ad000), int(0x148386000000), int(0x14874b2b7000)],
                                function_name = "H_matmul1")
        self.H_gelu = GeLU(data_type,
                                        addresses = [int(0x14acc3e2bc00), int(0x14acc3e5dc00)],
                                        function_name = "gelu")
        self.H_matmul2 = Matmul(data_type, function_name = "H_matmul2")
        self.layer_norm1= LayerNorm(data_type,
                                        addresses = [int(0x14acc3e09800), 0, 0],
                                        function_name = "layer_norm1")
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, _, d = x.shape
        assert d == self.d_model
        s = seq_len
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h

        # KV cache
        K_cache = Tensor([b, h // dev_cnt, d_h, s], self.data_type)
        V_cache = Tensor([b, h // dev_cnt, s, d_h], self.data_type)

        # multi-head attention
        q = self.Q_proj(x, self.Wq)  # [b, 1, d / dev_cnt]
        assert q.shape == [b, 1, d // dev_cnt]
        k = self.K_proj(x, self.Wk)  # [b, 1, d / dev_cnt]
        v = self.V_proj(x, self.Wv)  # [b, 1, d / dev_cnt]

        q = self.Q_reshape(q, [b, 1, h // dev_cnt, d_h])
        k = self.K_reshape(k, [b, 1, h // dev_cnt, d_h])
        v = self.V_reshape(v, [b, 1, h // dev_cnt, d_h])

        q_T = self.Q_transpose(q, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
        assert q_T.shape == [b, h // dev_cnt, 1, d_h]
        k_T = self.K_transpose(k, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, 1]
        assert k_T.shape == [b, h // dev_cnt, d_h, 1]
        v_T = self.V_transpose(v, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
        assert v_T.shape == [b, h // dev_cnt, 1, d_h]
        K_T = self.K_concat(K_cache, k_T, 3)  # [b, h / dev_cnt, d_h, s+1]
        assert K_T.shape == [b, h // dev_cnt, d_h, s + 1]
        V_T = self.V_concat(V_cache, v_T, 2)  # [b, h / dev_cnt, s+1, d_h]
        assert V_T.shape == [b, h // dev_cnt, s + 1, d_h]

    
        a = self.Q_mul_K(q_T, K_T)  # [b, h / dev_cnt, 1, s+1]
        assert a.shape == [b, h // dev_cnt, 1, s + 1]
        a_prob = self.A_softmax(a)
        h0 = self.A_mul_V(a_prob, V_T)  #  [b, h / dev_cnt, 1, d_h]
        assert h0.shape == [b, h // dev_cnt, 1, d_h]
        h0 = self.H_transpose(h0, [0, 2, 1, 3])  #  [b, 1, h / dev_cnt, d_h]
        assert h0.shape == [b, 1, h // dev_cnt, d_h]
        h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
        assert h0.shape == [b, 1, d // dev_cnt]

        h0 = self.H_matmul0(h0, self.W0)  #  [b, 1, d]
        assert h0.shape == [b, 1, d]
        h0 = self.layer_norm0(h0)
        assert h0.shape == [b, 1, d]
        if dev_cnt > 1:
            h0 = self.allreduce_mha(h0)

        # feed-forward network
        h1 = self.H_matmul1(h0, self.W1)  # [b, 1, 4 * d / dev_cnt]
        assert h1.shape == [b, 1, 4 * d // dev_cnt]
        h1 = self.H_gelu(h1)
        h2 = self.H_matmul2(h1, self.W2)  #  [b, 1, d]
        assert h2.shape == [b, 1, d]
        h2 = self.layer_norm1(h2)
        if dev_cnt > 1:
            h2 = self.allreduce_ffn(h2)

        assert h2.shape == [b, 1, d]
        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
            + self.W1.size * self.W1.data_type.word_size
            + self.W2.size * self.W2.data_type.word_size
            + K_cache.size * K_cache.data_type.word_size
            + V_cache.size * V_cache.data_type.word_size
        )
        return h2

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        qkv_latency = 3 * (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.roofline_latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        # print(f'memory requirement: {self.memory_requirement/1e9*96}GB')
        self.roofline_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.roofline_latency

    def compile_and_simulate(self, system: System, compile_mode: str):
        device = system.device
        interconnect = system.interconnect

        # matmul
        print("simulating q")
        q_latency = (
            self.Q_proj.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        print("simulating k")
        k_latency = (
            self.K_proj.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        print("simulating v")
        v_latency = (
            self.V_proj.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        print("simulating q_mul_k")
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )

        softmax_latency = (
            self.A_softmax.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.softmax
        )

        print("simulating a_mul_v")
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        print("simulating h_matmul0")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        layernorm0_latency = (
            self.layer_norm0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )

        print("simulating h1_matmul1")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )

        # gelu
        gelu_latency = (
            self.H_gelu.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.gelu
        )

        print("simulating h2_matmul2")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )

        layernorm1_latency = (
            self.layer_norm1.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )

        print("finish matmul simulation")

        matmul_total_latency = (
            q_latency
            + k_latency
            + v_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        normlization_total_latency = softmax_latency + layernorm0_latency + layernorm1_latency

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        # print("breakdown:")
        # print(
        #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        # )
        # print("total:")
        # print(
        #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        # )
        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        self.simluate_log = ""
        #f"{q_latency}, {k_latency},{v_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.latency

    def run_on_gpu(self):
        # matmul
        qkv_latency = (
            self.Q_proj.run_on_gpu()  # - self.Q_proj.gpu_kernel_launch_overhead()
        ) * 3
        q_mul_k_latency = (
            self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
        )
        a_mul_v_latency = (
            self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
        )
        h_matmul0_latency = (
            self.H_matmul0.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
        )
        h1_matmul1_latency = (
            self.H_matmul1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
        )
        h2_matmul2_latency = (
            self.H_matmul2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
        )
        layernorm_latency = (
            self.layer_norm0.run_on_gpu()
            - self.layer_norm0.gpu_kernel_launch_overhead()
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.run_on_gpu()  # - self.H_gelu.gpu_kernel_launch_overhead()
        )
        # gelu_latency = max(gelu_latency, 1e-7)

        # allreduce
        allreduce_total_latency = 0

        # others

        # print
        print("breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.latency_on_gpu = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        return self.latency_on_gpu


class LLMInitComputationTP:
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        device_count,
    ) -> None:
        pass


# # === GQA Configuration ===
# kv_heads = 8                            # Fixed for LLaMA 3–8B
# q_heads = self.n_heads                 # Typically 32
# d_h = self.d_model // q_heads          # e.g. 4096 / 32 = 128
# kv_d_h = self.d_model // kv_heads      # e.g. 4096 / 8 = 512

# # === KV Cache Initialization ===
# K_cache = Tensor([b, kv_heads // dev_cnt, kv_d_h, s], self.data_type)
# V_cache = Tensor([b, kv_heads // dev_cnt, s, kv_d_h], self.data_type)

# # === Linear Projections ===
# q = self.Q_proj(x, self.Wq)  # [b, 1, d // dev_cnt]
# assert q.shape == [b, 1, d // dev_cnt]

# k = self.K_proj(x, self.Wk)  # [b, 1, kv_heads * kv_d_h]
# v = self.V_proj(x, self.Wv)

# # === Reshape Q to [B, 1, q_heads, d_h] ===
# q = self.Q_reshape(q, [b, 1, q_heads // dev_cnt, d_h])

# # === Reshape K/V to [B, 1, kv_heads, kv_d_h] ===
# k = self.K_reshape(k, [b, 1, kv_heads // dev_cnt, kv_d_h])
# v = self.V_reshape(v, [b, 1, kv_heads // dev_cnt, kv_d_h])

# # === Broadcast K/V to match Q head count ===
# repeat_factor = q_heads // kv_heads   # 32 / 8 = 4
# k = k.repeat_interleave(repeat_factor, dim=2)  # [B, 1, q_heads, kv_d_h]
# v = v.repeat_interleave(repeat_factor, dim=2)

# # === Transpose to match attention input format ===
# q_T = self.Q_transpose(q, [0, 2, 1, 3])  # [b, q_heads, 1, d_h]
# k_T = self.K_transpose(k, [0, 2, 3, 1])  # [b, q_heads, d_h, 1]
# v_T = self.V_transpose(v, [0, 2, 1, 3])  # [b, q_heads, 1, d_h]

# # === KV Cache Concat ===
# K_T = self.K_concat(K_cache, k_T, 3)     # [b, q_heads, kv_d_h, s+1]
# V_T = self.V_concat(V_cache, v_T, 2)     # [b, q_heads, s+1, kv_d_h]

# # === Attention ===
# a = self.Q_mul_K(q_T, K_T)               # [b, q_heads, 1, s+1]
# a_prob = self.A_softmax(a)
# h0 = self.A_mul_V(a_prob, V_T)           # [b, q_heads, 1, d_h]

# # === Post-processing: reshape back ===
# h0 = self.H_transpose(h0, [0, 2, 1, 3])             # [b, 1, q_heads, d_h]
# h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])       # [b, 1, d]


class TransformerBlockAutoRegressionTPGQA(Operator):
    def __init__(self, d_model, n_heads, num_kv_heads, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = num_kv_heads
        self.device_count = device_count

        d = d_model
        d_h = d // n_heads
        kv_h = num_kv_heads
        kv_dim = d_h * kv_h // device_count

        # Parameters
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, kv_dim], data_type)
        self.Wv = Tensor([d, kv_dim], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)

        # Operators
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)
        self.K_repeat = RepeatInterleave(data_type)  # New operator
        self.V_repeat = RepeatInterleave(data_type)  # New operator
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)

        self.H_matmul1 = Matmul(data_type, addresses = [int(0x14874b2ad000), int(0x148386000000), int(0x14874b2b7000)])
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        b, _, d = x.shape
        assert d == self.d_model
        s = seq_len
        h = self.n_heads
        kv_h = self.n_kv_heads
        dev_cnt = self.device_count
        d_h = d // h

        # KV cache
        K_cache = Tensor([b, kv_h // dev_cnt, d_h, s], self.data_type)
        V_cache = Tensor([b, kv_h // dev_cnt, s, d_h], self.data_type)

        # Project Q, K, V
        q = self.Q_proj(x, self.Wq)
        k = self.K_proj(x, self.Wk)
        v = self.V_proj(x, self.Wv)

        # Reshape
        q = self.Q_reshape(q, [b, 1, h // dev_cnt, d_h])
        k = self.K_reshape(k, [b, 1, kv_h // dev_cnt, d_h])
        v = self.V_reshape(v, [b, 1, kv_h // dev_cnt, d_h])

        # Repeat KV to match Q heads
        repeat_factor = (h // dev_cnt) // (kv_h // dev_cnt)
        k = self.K_repeat(k, repeat_factor, axis=2)
        v = self.V_repeat(v, repeat_factor, axis=2)

        # Transpose and attention
        q_T = self.Q_transpose(q, [0, 2, 1, 3])
        k_T = self.K_transpose(k, [0, 2, 3, 1])
        v_T = self.V_transpose(v, [0, 2, 1, 3])
        K_T = self.K_concat(K_cache, k_T, 3)
        V_T = self.V_concat(V_cache, v_T, 2)
        a = self.Q_mul_K(q_T, K_T)
        a_prob = self.A_softmax(a)
        h0 = self.A_mul_V(a_prob, V_T)

        # Output reshaping
        h0 = self.H_transpose(h0, [0, 2, 1, 3])
        h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
        h0 = self.H_matmul0(h0, self.W0)
        h0 = self.layer_norm0(h0)
        if dev_cnt > 1:
            h0 = self.allreduce_mha(h0)

        # FFN
        h1 = self.H_matmul1(h0, self.W1)
        h1 = self.H_gelu(h1)
        h2 = self.H_matmul2(h1, self.W2)
        h2 = self.layer_norm1(h2)
        if dev_cnt > 1:
            h2 = self.allreduce_ffn(h2)

        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size +
            self.Wk.size * self.Wk.data_type.word_size +
            self.Wv.size * self.Wv.data_type.word_size +
            self.W0.size * self.W0.data_type.word_size +
            self.W1.size * self.W1.data_type.word_size +
            self.W2.size * self.W2.data_type.word_size +
            K_cache.size * K_cache.data_type.word_size +
            V_cache.size * V_cache.data_type.word_size
        )
        return h2


class TransformerBlockAutoRegressionTPMQA(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = 1  # MQA: only 1 key/value head
        self.device_count = device_count

        d = d_model
        d_h = d // n_heads
        kv_dim = d_h * self.n_kv_heads // device_count

        # Parameters
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, kv_dim], data_type)
        self.Wv = Tensor([d, kv_dim], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)

        # Operators
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)
        self.K_repeat = RepeatInterleave(data_type)
        self.V_repeat = RepeatInterleave(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)

        self.H_matmul1 = Matmul(data_type, addresses=[int(0x14874b2ad000), int(0x148386000000), int(0x14874b2b7000)])
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        b, _, d = x.shape
        assert d == self.d_model
        s = seq_len
        h = self.n_heads
        kv_h = self.n_kv_heads
        dev_cnt = self.device_count
        d_h = d // h

        # KV cache
        K_cache = Tensor([b, kv_h // dev_cnt, d_h, s], self.data_type)
        V_cache = Tensor([b, kv_h // dev_cnt, s, d_h], self.data_type)

        # Project Q, K, V
        q = self.Q_proj(x, self.Wq)
        k = self.K_proj(x, self.Wk)
        v = self.V_proj(x, self.Wv)

        q = self.Q_reshape(q, [b, 1, h // dev_cnt, d_h])
        k = self.K_reshape(k, [b, 1, kv_h // dev_cnt, d_h])
        v = self.V_reshape(v, [b, 1, kv_h // dev_cnt, d_h])

        # Repeat KV to match Q heads
        repeat_factor = (h // dev_cnt) // (kv_h // dev_cnt)
        k = self.K_repeat(k, repeat_factor, axis=2)
        v = self.V_repeat(v, repeat_factor, axis=2)

        q_T = self.Q_transpose(q, [0, 2, 1, 3])
        k_T = self.K_transpose(k, [0, 2, 3, 1])
        v_T = self.V_transpose(v, [0, 2, 1, 3])
        K_T = self.K_concat(K_cache, k_T, 3)
        V_T = self.V_concat(V_cache, v_T, 2)
        a = self.Q_mul_K(q_T, K_T)
        a_prob = self.A_softmax(a)
        h0 = self.A_mul_V(a_prob, V_T)

        h0 = self.H_transpose(h0, [0, 2, 1, 3])
        h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
        h0 = self.H_matmul0(h0, self.W0)
        h0 = self.layer_norm0(h0)
        if dev_cnt > 1:
            h0 = self.allreduce_mha(h0)

        h1 = self.H_matmul1(h0, self.W1)
        h1 = self.H_gelu(h1)
        h2 = self.H_matmul2(h1, self.W2)
        h2 = self.layer_norm1(h2)
        if dev_cnt > 1:
            h2 = self.allreduce_ffn(h2)

        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size +
            self.Wk.size * self.Wk.data_type.word_size +
            self.Wv.size * self.Wv.data_type.word_size +
            self.W0.size * self.W0.data_type.word_size +
            self.W1.size * self.W1.data_type.word_size +
            self.W2.size * self.W2.data_type.word_size +
            K_cache.size * K_cache.data_type.word_size +
            V_cache.size * V_cache.data_type.word_size
        )
        return h2
