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

trace = TraceLogger.instance()

# model_init = TransformerBlockInitComputationTP(
#     d_model=12288,
#     n_heads=96,
#     device_count=device_count,
#     data_type=data_type_dict["fp16"],
# )
model_auto_regression = TransformerBlockAutoRegressionTP(
    d_model=4096,
    n_heads=32,
    device_count=device_count,
    data_type=data_type_dict["fp16"],
)
# _ = model_init(
#     Tensor([batch_size, input_seq_length, model_init.d_model], data_type_dict["fp16"])
# )
_ = model_auto_regression(
    Tensor([batch_size, device_count, model_auto_regression.d_model], data_type_dict["fp16"]),
    input_seq_length + output_seq_length,
)
auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
	system, "heuristic-GPU"
)

print(model_auto_regression.simluate_log)
trace.save_trace()