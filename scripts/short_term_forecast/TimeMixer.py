import subprocess
import os

# 设置 CUDA 可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 定义通用参数
model_name = "TimeMixer"
e_layers = 4
down_sampling_layers = 1
down_sampling_window = 2
learning_rate = 0.01
d_model = 32

# 定义不同的季节性模式及其对应的 d_ff 值
seasonal_patterns = {
    'Monthly': 32,
    'Yearly': 32,
    'Quarterly': 64,
    'Daily': 16,
    'Weekly': 32,
    'Hourly': 32
}

# 遍历不同的季节性模式
for pattern, d_ff in seasonal_patterns.items():
    model_id = f"m4_{pattern}"
    command = [
        "python", "-u", "run.py",
        "--task_name", "short_term_forecast",
        "--is_training", "1",
        "--root_path", "./dataset/m4",
        "--seasonal_patterns", pattern,
        "--model_id", model_id,
        "--model", model_name,
        "--data", "m4",
        "--features", "M",
        "--e_layers", str(e_layers),
        "--d_layers", "1",
        "--factor", "3",
        "--enc_in", "1",
        "--dec_in", "1",
        "--c_out", "1",
        "--batch_size", "128",
        "--d_model", str(d_model),
        "--d_ff", str(d_ff),
        "--des", "Exp",
        "--itr", "1",
        "--learning_rate", str(learning_rate),
        "--train_epochs", "50",
        "--patience", "20",
        "--down_sampling_layers", str(down_sampling_layers),
        "--down_sampling_method", "avg",
        "--down_sampling_window", str(down_sampling_window),
        "--loss", "SMAPE"
    ]

    try:
        # 执行命令
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"命令执行出错: {e}")
    