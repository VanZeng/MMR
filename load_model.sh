#!/bin/bash

# 脚本名称：download_model.sh
# 功能：从 Hugging Face 下载指定模型

# 定义参数
MODEL_NAME="nyu-visionx/cambrian-phi3-3b"
LOCAL_DIR="cambrian-phi3-3b"

# 执行下载命令
mkdir $LOCAL_DIR
echo "开始下载模型 $MODEL_NAME ..."
huggingface-cli download --resume-download $MODEL_NAME --local-dir $LOCAL_DIR

# 检查是否成功
if [ $? -eq 0 ]; then
  echo "✅ 下载完成！模型已保存到目录: $LOCAL_DIR"
else
  echo "❌ 下载失败，请检查网络或参数！"
fi