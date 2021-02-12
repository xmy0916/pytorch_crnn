# 项目简介
本项目基于Pytorch实现CRNN文字识别项目，从零重写项目，学习CRNN实现的细节。

# 项目参考
论文地址：[https://arxiv.org/abs/1507.05717](https://arxiv.org/abs/1507.05717)

参考代码：[https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec](https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec)

# 数据集
本文使用的是MLT2017的数据集，下载地址：[https://download.csdn.net/download/qq_37668436/15045746](https://download.csdn.net/download/qq_37668436/15045746)


# 博客地址
[从零写CRNN](https://blog.csdn.net/qq_37668436/article/details/113642808)

# 项目运行
- 修改config/config.yml文件中的CHAR_FILE和JSON_FILE参数到你的数据集路径
- python3 train.py --cfg config/config.yml 运行训练
