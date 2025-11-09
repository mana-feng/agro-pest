# Structure
agro-pest/
├── config.yaml          # 配置（路径 & 训练参数）
├── requirements.txt     # 依赖
├── README.md
├── src/
│   └── gp_resnet18.py   # 训练 + 评测主脚本
└── data/
└── archive/         # 放数据集（含 train/valid/test）

data/archive/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/

pip install -r requirements.txt

Output:
curve_loss.png — training loss curve
curve_metrics.png — validation accuracy and F1 curve
cm_test.png — test confusion matrix
classification_report_test.txt — precision/recall/F1 per class