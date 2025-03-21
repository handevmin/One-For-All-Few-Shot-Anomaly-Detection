# One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning

Official implement of [One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning](https://openreview.net/forum?id=Zzs3JwknAY)

![framework](https://github.com/Vanssssry/One-For-All-Few-Shot-Anomaly-Detection/blob/main/framework.png)



## Installation

```shell
conda create -n IIPAD
conda activate IIPAD
pip install -r requirements.txt
```



## Data

Download the dataset from [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad)

Download the dataset from [VisA](https://github.com/amazon-science/spot-diff?tab=readme-ov-file#data-download)

​	

## Training 

Please view the `test.sh` file and run.

```bash
bash test.sh
```

The main program is implemented in `main.py`.



## Citation

Please cite the following paper if this work helps your project:

```
@inproceedings{
lv2025oneforall,
title={One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning},
author={Wenxi Lv and Qinliang Su and Wenchao Xu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=Zzs3JwknAY}
}
```

## Acknowledge

We thank the great works [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) and [AnoVL](https://github.com/hq-deng/AnoVL) for assisting with our work.