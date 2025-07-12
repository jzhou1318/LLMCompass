from utils import TraceLogger
from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.utils import data_type_dict, Tensor
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)

input_seq_length = 32
batch_size = 1
output_seq_length = 4
arch_specs = read_architecture_template("configs/template.json")
system = template_to_system(arch_specs)
device_count = arch_specs["device_count"]

# SRAM
# "SRAM_KB": 192
# pcb_module.compute_module.core.SRAM_size 
# core_specs["SRAM_KB"] * 1024,

# L2 Size
# "global_buffer_MB": 40,
# io_specs["global_buffer_MB"] * 1024 * 1024,
# pcb_module.compute_module.l2_size

# "physical_global_buffer_MB": 48

trace = TraceLogger.instance()

# model_init = TransformerBlockInitComputationTP(
#     d_model=12288,
#     n_heads=96,
#     device_count=device_count,
#     data_type=data_type_dict["fp16"],
# )
# model_auto_regression = TransformerBlockAutoRegressionTP(
#     d_model=4096,
#     n_heads=32,
#     device_count=device_count,
#     data_type=data_type_dict["fp16"],
# )
model_auto_regression = TransformerBlockAutoRegressionTP(
    d_model=1600,
    n_heads=25,
    device_count=device_count,
    data_type=data_type_dict["fp16"],
)
_ = model_auto_regression(
    Tensor([batch_size, 1, model_auto_regression.d_model], data_type_dict["fp16"]),
    input_seq_length + output_seq_length,
)
auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
	system, "heuristic-GPU"
)

print(model_auto_regression.simluate_log)
trace.save_trace()

trace.coalesce(filepath = 'memory_trace.csv')