# Structure
agro-pest/
├── config.yaml
├── requirements.txt
├── README.md
├── src/
│   └── gp_resnet18.py
└── data/
└── archive/

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