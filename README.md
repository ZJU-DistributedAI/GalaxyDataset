# Produce DAI dataset on CIFAR10
## requirements for daidataset

version 2: 

According to number of nodes, we split CIFAR10 dataset. We can customize the config.yaml configuration file. Set the split mode, randomly or by category. Set the number of nodes, the size of the data set for each node, and the number of each node category. 
In addition, we can also set the same distribution of data sets to increase the error dataset.

1. randomSplit: 
    1) no error dataset 
    2) add error dataset
2. splitByLabel: 
    1. just split dataset
    2. add other dataset, no error dataset
    3. add error dataset, no other dataset 
    4. add both dataset

Adding Standing: This version adds a standard to measure the degree of non independence and distribution of data. Using NI,

Given a feature extractor $g_{\varphi}(\cdot)$ and a class $C$, the degree of distribution shift is $D_{test}^{C}$   deﬁned as:

$N I(C)=\left\|\frac{\overline{g_{\varphi}\left(X_{\text {train}}^{C}\right)}-\overline{g_{\varphi}\left(X_{\text {test}}^{C}\right)}}{\sigma\left(g_{\varphi}\left(X^{C}\right)\right)}\right\|_{2}$

where$X^{C}=X_{\text {train}}^{C} \cup X_{\text {test}}^{C}, \overline{(\cdot)}$ represents the ﬁrst order moment, $\sigma(\cdot)$ is the std used to

normalize the scale of features and $\|\cdot\|_{2}$epresents the 2-norm.

## Usage
Environment: python3.6
```
pip3 install -r requirements.txt
```
step1:  `source ~/venv/bin/activate`

step2: set config.yaml

step3: `./splitDataset.sh`

## Parameter

In config.yaml file: we will create non-iid dataset based on the number of nodes;

We will set  node_num  parameter which represents the number of dataset we want to generate. For example, we want 500 dataset with 500nodes. We will set node_num: 500.

```
dataset_mode: randomly or by category;
node_num: the number of nodes
dataset_size_list: the size of the data set for each node
node_label_num: the number of each node category
isuse_yaml: True=> run main.py by yaml.file, Flase=> run main.py by command line.
```
We will generate n custom datasets.
On cifar10 files, every npy file consists of python's List.

```
npy file: [[imgs, label], [imgs, label]...., [imgs, label]]
```
We can see the name of each npy file:
```
npy file name: split node index mode_dataset size_target label_the ratio of other label_the ratio of error dataset
```
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
dataloader = readnpy("./cifar10/splitByLabelsWithNormalAndErrorDataset/SplitByLabels_2222_horseandMore_0.1_0.01.npy")
```