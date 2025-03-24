import subprocess

# 设置使用的 GPU
cuda_visible_devices = "0"
model_name = "TimesNet"

# 定义不同的季节性模式及其对应的参数
seasonal_patterns = [
    {
        "pattern": "Monthly",
        "model_id": "m4_Monthly",
        "d_model": 32,
        "d_ff": 32
    }
]

# 设置环境变量
import os
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

# 循环执行不同季节性模式的训练任务
for pattern_info in seasonal_patterns:
    command = [
        "python", "-u", "run.py",
        "--task_name", "short_term_forecast",
        "--is_training", "1",
        "--root_path", "./dataset/m4",
        "--seasonal_patterns", pattern_info["pattern"],
        "--model_id", pattern_info["model_id"],
        "--model", model_name,
        "--data", "m4",
        "--features", "M",
        "--e_layers", "2",
        "--d_layers", "1",
        "--factor", "3",
        "--enc_in", "1",
        "--dec_in", "1",
        "--c_out", "1",
        "--batch_size", "16",
        "--d_model", str(pattern_info["d_model"]),
        "--d_ff", str(pattern_info["d_ff"]),
        "--top_k", "5",
        "--des", "Exp",
        "--itr", "1",
        "--learning_rate", "0.001",
        "--loss", "SMAPE",
        "--use_gpu", "True"
    ]
    try:
        # 执行命令
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")


# import torch

# torch_version = torch.__version__
# print(f"PyTorch 版本: {torch_version}")

# if torch.cuda.is_available():
#     cuda_version = torch.version.cuda
#     print(f"PyTorch 使用的 CUDA 版本: {cuda_version}")
# else:
#     print("当前环境不支持 CUDA。")