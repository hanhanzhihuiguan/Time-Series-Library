import subprocess

# 设置 CUDA 可见设备
cuda_visible_devices = "0"
model_name = "iTransformer"

# 定义不同的预测长度
pred_lens = [96, 192, 336, 720]

# 定义通用的命令参数
common_args = [
    "--task_name", "long_term_forecast",
    "--is_training", "1",
    "--root_path", "./dataset/weather/",
    "--data_path", "weather.csv",
    "--model", model_name,
    "--data", "custom",
    "--features", "M",
    "--seq_len", "96",
    "--label_len", "48",
    "--e_layers", "3",
    "--d_layers", "1",
    "--factor", "3",
    "--enc_in", "21",
    "--dec_in", "21",
    "--c_out", "21",
    "--des", "Exp",
    "--d_model", "512",
    "--d_ff", "512",
    "--itr", "1"
]

# 设置环境变量
env = {"CUDA_VISIBLE_DEVICES": cuda_visible_devices}

for pred_len in pred_lens:
    # 生成 model_id
    model_id = f"weather_96_{pred_len}"

    # 构建完整的命令
    command = ["python", "-u", "run.py"] + common_args + [
        "--model_id", model_id,
        "--pred_len", str(pred_len)
    ]

    try:
        # 执行命令
        print(f"正在运行命令: {' '.join(command)}")
        subprocess.run(command, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"命令执行出错: {e}")
