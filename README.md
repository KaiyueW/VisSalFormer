# SalChartQA: Question-driven Saliency on Information Visualisations

[![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--3884-d45815.svg)](https://doi.org/10.18419/darus-3884)

*Yao Wang, Weitian Wang, Abdullah Abdelhafez, Mayar Elfares, Zhiming Hu, Mihai Bâce, and Andreas Bulling*

Proceedings of the ACM SIGCHI Conference on Human Factors in Computing Systems (CHI 2024)


```
$Root Directory
│
│─ README.md —— this file
│
|─ Code —— Source code of the VisSalFormer model to predict question-driven saliency
│  │
│  |─ environment.yml —— conda environment
│  │
│  |─ env.py —— python envorinment $TORCH_HOME and $TRANSFORMERS_CACHE 
│  │
│  │─ dataset_new.py —— dataloader for SalChartQA
│  │
│  │─ evaluation.py —— evaluation script to load VisSalFormer weights and make predictions
│  │
│  │─ evaluation.sh —— bash script to run evaluation.py
│  │
│  │─ model_swin.py —— definition of the VisSalFormer model
│  │
│  │─ tokenizer_bert.py —— tokenizer of Bert
│  │
│  └─ VisSalFormer_weights.tar —— weights of VisSalFormer
│
└─ SalChartQA.zip —— The SalChartQA dataset
   │
   │─ fixationByVis —— BubbleView data (mouse clicks) of AMT workers
   │
   │─ image_questions.json —— visualisation-question pairs
   │
   │─ raw_img —— original visualisations from the ChartQA dataset
   │
   │─ saliency_all —— saliency maps from all AMT workers
   │
   │─ saliency_ans —— saliency maps aggretated by all AMT workers who either answered a question correctly or wrongly
   │
   └─ unified_approved.csv —— responses from AMT workers

```

If you think our work is useful to you, please consider citing our paper as:

```
@inproceedings{wang24_chi,
  title = {SalChartQA: Question-driven Saliency on Information Visualisations},
  author = {Wang, Yao and Wang, Weitian and Abdelhafez, Abdullah and Elfares, Mayar and Hu, Zhiming and B{\^a}ce, Mihai and Bulling, Andreas},
  year = {2024},
  pages = {1--14},
  booktitle = {Proc. ACM SIGCHI Conference on Human Factors in Computing Systems (CHI)},
  doi = {10.1145/3613904.3642942}
}

```


contact: yao.wang@vis.uni-stuttgart.de