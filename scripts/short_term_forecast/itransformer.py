import subprocess
import os

# 设置 CUDA 可见设备
cuda_visible_devices = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

# 定义模型名称
model_name = "iTransformer"

# 定义不同的预测长度
pred_lens = [96, 192, 336, 720]

# 循环执行不同预测长度的训练任务
for pred_len in pred_lens:
    model_id = f"weather_96_{pred_len}"
    command = [
        "python", "-u", "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", "./dataset/weather/",
        "--data_path", "weather.csv",
        "--model_id", model_id,
        "--model", model_name,
        "--data", "custom",
        "--features", "M",
        "--seq_len", "96",
        "--label_len", "48",
        "--pred_len", str(pred_len),
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
    try:
        # 执行命令
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"命令执行出错: {e}")