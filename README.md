# Few-Shot Graph and SMILES Learning for Molecular Property Prediction

## Introduction
This is the source code and dataset for the following paper: 

**Few-Shot Graph and SMILES Learning for Molecular Property Prediction**

Contact Dan Sun (2201793@s.hlju.edu.cn,), if you have any questions.

## Datasets
The datasets uploaded can be downloaded to train our model directly.

The original datasets are downloaded from [Data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip). We utilize Original_datasets/splitdata.py to split the datasets according to the molecular properties and save them in different files in the Original_datasets/[DatasetName]/new. Then run main.py, the datasets will be automatically preprocessed by loader.py and the preprocessed results will be saved in the Original_datasets/[DatasetName]/new/[PropertyNumber]/propcessed.

## Usage

### Installation
We used the following Python packages for the development by python 3.6.
```
- torch = 1.4.0
- torch-geometric = 1.6.1
- torch-scatter = 2.0.4
- torch-sparse = 0.6.1
- scikit-learn = 0.23.2
- tqdm = 4.50.0
- rdkit
```

### Run code

Datasets and k (for k-shot) can be changed in the last line of main.py.
```
python main.py
```

## Performance
The performance of meta-learning is not stable for some properties. We report two times results and the number of the iteration where we obtain the best results here for your reference.

| Dataset    | k    | Iteration | Property   | Results   || k    | Iteration | Property  | Results   |
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------:  | ---------- | :-----------:  | :-----------: | :-----------: | :-----------:  |
| Sider | 1 | 307/599 | Si-T1| 74.15/74.27 | | 5 | 561/585 | Si-T1 | 75.16/75.52 | 
|  |  | | Si-T2| 68.34/68.85 | |  | | Si-T2 | 68.90/70.06 | 
|  |  | | Si-T3| 69.90/70.13 | |  | | Si-T3 | 72.03/72.04 | 
|  |  | | Si-T4| 71.78/71.88 | |  | | Si-T4 | 72.40/72.51 | 
|  |  | | Si-T5| 78.40/78.72 | |  | | Si-T5 | 79.71/79.86 | 
|  |  | | Si-T6| 69.59/70.44 | |  | | Si-T6 | 71.90/72.33 | 
|  |  | | Ave.| 72.03/72.38 | |  | | Ave. | 73.35/73.72 | 
| Tox21 | 1 | 1271/1415 | SR-HS | 74.27/74.86 | | 5 | 1061/882 | SR-HS | 74.85/75.24 | 
|  |  | | SR-MMP | 79.62/80.06 | |  | | SR-MMP | 80.10/80.15 | 
|  |  | | SR-p53| 77.91/78.87 | |  | | SR-p53 | 78.86/79.33 | 
|  |  | | Ave.| 77.27/77.93 | |  | | Ave. | 77.94/78.24 | 

## Acknowledgements

The code is implemented based on [Few-shot Graph Learning for Molecular Property Prediction](https://github.com/zhichunguo/Meta-MGNN).

## Reference

```
@article{
  title={Few-Shot Graph and SMILES Learning for Molecular Property Prediction},
  author={Dan, Sun and Yong, Liu and Zhang, Wei},
  year={2022}
}
```
