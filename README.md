# MMR：Multimodal Large Language Model Robustness Test



这是一个测试代码，用来对噪声进行分级，选用高斯噪声。cambrian是抄过来的代码。

**建立环境**

```
conda create -n cambrian python==3.10
conda activate cambrian
```

**安装依赖**

```
pip install -r requirements.txt
```

**下载cam-3b模型**

```
# 穷b只能跑3b模型
chmod a+x ./load_model.sh
./load_model.sh # 不会开代理可以用hf-mirror走镜像
```

**运行测试**

运行过程中会安装一些cambrian的编码器、解码器，不要惊慌，正常下载即可，只用下载一次。

```
python3 rank.py
```

运行过程中，你会看到生成了一些文件，results里存的是不同噪声强度的测试结果，incorrect存档模型回答错误的问题，backup是备份文件夹，answers存的是模型的答案，metadata.csv是元数据。