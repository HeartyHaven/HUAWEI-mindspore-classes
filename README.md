# 华为昇腾目标检测实战课程
HAVEN Implemented this on 28/10/2023
## Introduction
华中科技大学与华为公司合作开展的华为昇腾目标检测实战课程，这个课主要是使用`MindSpore`库完成一些机器学习和深度学习的实践内容。

`MindSpore`的语法有些类似`Keras`，不过要简洁一些，并且支持华为原生的昇腾计算框架。缺点是支持的CUDA版本比较有限，而且要想熟练运用可能还得了解下华为云计算平台的型号等等，否则如果选用了错误的服务器架构，有些函数可能不能用，或者参数选择上有限制。

主要工作是将课程中提供的代码进行了复现。

内容截至目前包含：
- 基于MindSpore的手写体识别；
- 基于MindSpore的目标检测(YOLOv3)；
- MNIST手写体识别-CANN推理；
- 基于MindX SDK开发目标检测应用。
