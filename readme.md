# GENET

GENET: Unleashing the Power of Side Information for Recommendation via Hypergraph Pretraining

### Code structure

```
.
├── config
│   ├── finetune.json
│   ├── finetune_seq.json
│   └── pretrain.json
├── data
│   ├── amazon.zip
│   ├── dataset.py
│   ├── dataset_wrapers.py
│   └── process_amazon.py
├── finetune
│   ├── finetuning_fewshots.py
│   ├── finetuning_gnn_signal.py
│   ├── finetuning_item_coldstart.py
│   ├── finetuning_user_coldstart.py
│   └── no_tuning.py
├── finetune_seq
│   └── finetuning_seq_signal.py
├── models
│   ├── HGNNP.py
│   └── Signal.py
├── pretrain
│   └── pretrain_lp_pm_plus.py
├── readme.md
├── save
│   ├── model
│   │   └── amazon
│   │       └── HGNNP_plus_LP_PM_500.pth
│   └── structure
│       └── amazon
│           └── hg_plus.pkl
└── utils
    ├── amazon.py
    ├── batch_test.py
    ├── cold_start.py
    ├── foursquare.py
    ├── gowalla.py
    ├── metrics.py
    └── visualize.py
```

### Requirements

```
pip install -r requirements.txt         # Install requirements with pip
```
### Datasets
We have provided the preprocessed Books datasets in the data/ folder. Please unzip the amazon.zip file before running the code.
We have also provided the pre-trained model in the save/model/ folder. Please unzip them.


### Quick start
```
python finetune/finetuning_gnn_signal.py   # Run a experiment, before that, please setup the root path of the dataset in the code.
```

Experiments can be controlled with the **./experiments/tpch.json** file. For descriptions of the components and functioning, consult our paper.
