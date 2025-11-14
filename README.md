Manual
==
请先确保你已经下载并正确path cuda，因为用cpu的速度真的很慢。  
将data文件夹放在和py文件同样的根目录下。  
接下来运行DenseNet.py，根据你设备的配置最高可能需要2小时训练完全部训练集。  


使用小一点的训练集
==
在剪裁完图片后=》出现run.dataset_cls文件夹，请确保每个文件夹都有图片或者txt。  
运行test.py对该文件夹进行过滤。  
默认设置是每类照片50个。  


densenet/
├── README.md
├── src/
│   └── densenet.py
└── data/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/


大概是可以做到
YOLO → Crop → ImageFolder 分类
- 数据增强（RandomResizedCrop, ColorJitter, Flip）
- 训练 + 验证 + 测试
- Test Confusion Matrix
- Classification Report
- Loss & F1 曲线
- 最优模型保存（best_densenet.pth）
