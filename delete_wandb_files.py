import os
import datetime
import shutil

def delete_wandb_folders(path):
    # 设定时间阈值：2024年2月1日
    cutoff_date = datetime.datetime(2024, 2, 1)

    # 遍历目录中的所有项目
    for item in os.listdir(path):
        full_path = os.path.join(path, item)

        # 确保是一个目录且目录名包含'wandb'
        if os.path.isdir(full_path) and 'run-' in item:
            # 获取目录的创建时间
            creation_time = os.path.getctime(full_path)
            folder_date = datetime.datetime.fromtimestamp(creation_time)

            # 如果创建时间早于阈值，删除该目录
            if folder_date < cutoff_date:
                print(f"Deleting {full_path}")
                # os.rmdir(full_path)  # 注意：只有当文件夹为空时，os.rmdir才能删除文件夹
                shutil.rmtree(full_path)

# 使用函数
delete_wandb_folders('./wandb')  # 替换为你的实际目录路径
