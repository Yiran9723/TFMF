# 大论文第二个方法（第三四章）&小论文模型
在Transformer基础上对顺序编码（第三章）进行改进，同时加入了空间交互模块和行驶意图模块进行多特征融合（第四章）。

## 评价指标
ADE(平均轨迹误差)&FDE（末点轨迹误差)

## 配置
* Python 3
* PyTorch (1.2)
* Matplotlib

## 数据集
NGSIM数据集，dataset文件夹里是处理后的数据，具体在文件夹里有说明。

## 运行
train.py训练模型；evaluate.py测试模型；draw_trajectory.py可以画轨迹图。

