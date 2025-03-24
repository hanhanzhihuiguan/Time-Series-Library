import subprocess
import os

# 设置使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 定义模型名称
model_name = "TimesNet"

# 定义不同的mask率及其对应的model_id
mask_configs = [
    {"mask_rate": 0.125, "model_id": "ETTh1_mask_0.125"},
    {"mask_rate": 0.25, "model_id": "ETTh1_mask_0.25"},
    {"mask_rate": 0.375, "model_id": "ETTh1_mask_0.375"},
    {"mask_rate": 0.5, "model_id": "ETTh1_mask_0.5"}
]

# 循环执行不同mask率的训练任务
for config in mask_configs:
    command = [
        "python", "-u", "run.py",
        "--task_name", "imputation",
        "--is_training", "1",
        "--root_path", "./dataset/ETT-small/",
        "--data_path", "ETTh1.csv",
        "--model_id", config["model_id"],
        "--mask_rate", str(config["mask_rate"]),
        "--model", model_name,
        "--data", "ETTh1",
        "--features", "M",
        "--seq_len", "96",
        "--label_len", "0",
        "--pred_len", "0",
        "--e_layers", "2",
        "--d_layers", "1",
        "--factor", "3",
        "--enc_in", "7",
        "--dec_in", "7",
        "--c_out", "7",
        "--batch_size", "16",
        "--d_model", "16",
        "--d_ff", "32",
        "--des", "Exp",
        "--itr", "1",
        "--top_k", "3",
        "--learning_rate", "0.001"
    ]
    try:
        # 执行命令并检查错误
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"执行命令时出错: {e}")
        # 如果需要在出错时停止后续任务，可以添加break