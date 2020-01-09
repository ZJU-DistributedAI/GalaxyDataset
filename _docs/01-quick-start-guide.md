---
title: "Quick-Start Guide"
permalink: /docs/quick-start-guide/
excerpt: "How to quickly install and setup galaxy dataset framework."
last_modified_at: 2019-08-20T21:36:11-04:00
redirect_from:
  - /theme-setup/
toc: true
---

# Evaluation metrics and a benchmark
We  propose a modular evaluation metrics and a benchmark for large-scale federated learning. Firstly, we construct a suite of open-source non-IID dataset by providing three methods including partitioning of datasets randomly, partitioning dataset with digit labels, and redefining datasets’ labels with both main concepts and contexts of datasets, which are grounded inreal-world assumptions. In addition, we design a rigorous evaluation metrics including the number of network nodes, the size of datasets, the number of communication rounds, communication resources, etc. Finally, we provide an open-source benchmark data for large-scale federated learning research.

## Installation

### Requirements

requirements.txt

```yaml
matplotlib==3.1.1
pandas==0.25.0
pyparsing==2.4.1.1
pyrsistent==0.15.3
python-dateutil==2.8.0
torch==1.2.0
torchvision==0.4.0
yaml
```

Environment: python3.6

```python
pip3 install -r requirements.txt
```

## Usage 

### For NEI values
step1: Download datasets

    python3 downloadData.py 
    

step2: run NEI.py with datasets

    python3 NEI.py 


step3: get results about NEI values

### For Non-IID Dataset Generation


In this part, we use `downloadData.py 、makeDataset.py and preprocess.py` to generate non-IID datasets. More importantly, we provide a config file config.yaml for setting related parameters about non-IID datasets. We now work on MNIST and CIFAR10 datasets.

```
python3 downloadData.py

python3 makeDataset.py 

python3 preprocess.py
```


### Setting config.yaml:

```
# dataset
dataset_mode: MNIST/ CIFAR10

# node num  Number of node (default n=10) one node corresponding to one dataset,
node_num: 4

# small dataset config
isaverage_dataset_size: false # if average splits dataset
# numpy.random.randint(10000, 20000 , 20)
#dataset_size_list: [18268, 16718, 10724, 12262, 17094, 14846, 17888, 14273, 13869,
##                    18087, 19631, 15390, 14346, 12743, 11246, 18375, 15813, 18280,
##                    12300, 12190]
dataset_size_list: [1822, 1000, 1200, 1300]

# split mode
split_mode: 0 # dataset split: randomSplit(0), splitByLabels(1)

# each node - label kind , sum must <= 10
# numpy.random.randint(0, 11, 20)
node_label_num: [ 4,  3,  2, 10]

isadd_label: false
add_label_rate: 0.1

isadd_error: false
add_error_rate: 0.01
```

We will generate n custom datasets.s

#### In downloadData.py:

We can download differents datasets.

#### In makeDataset.py:

In this part, we provide different methods for partitioning dataset.

1. randomSplit: 
    1) no error dataset 
    2) add error dataset
2. splitByLabel: 
    1. just split dataset
    2. add other dataset, no error dataset
    3. add error dataset, no other dataset 
    4. add both dataset
3. redefineLabels

We will generate n custom datasets.
On cifar10 files, every npy file consists of python's List.

```
npy file: [[imgs, label], [imgs, label]...., [imgs, label]]
```

We can see the name of each npy file:

```
npy file name: split node index mode_dataset size_target label_the ratio of other label_the ratio of error dataset
```

#### In preprocess.py:

We will use readnpy method to read npy file

```python
def readnpy(path):
    # npy file: [[imgs, label], [imgs, label]...., [imgs, label]]
    # when allow_pickle=True, matrix needs same size
    np_array = np.load(path, allow_pickle=True)
    imgs = []
    label = []
    for index in range(len(np_array)):
        imgs.append(np_array[index][0])
        label.append(np_array[index][1])
    torch_dataset = Data.TensorDataset(torch.from_numpy(np.array(imgs)), torch.from_numpy(np.array(label)))

    dataloader = Data.DataLoader(
        torch_dataset,
        batch_size=64,
        shuffle=True
    )
    print(dataloader)
    return dataloader
dataloader = readnpy("./XXXX/splitByLabelsWithNormalAndErrorDataset/SplitByLabels_2222_horseandMore_0.1_0.01.npy")
```
