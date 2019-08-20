# Produce DAI dataset on CIFAR10
> requirements for daidataset
1. Randomly split CIFAR10 into 10 small datasets
2. Divide CIFAR10 into 10 small datasets according to dataset labels
3. Based on the 2nd method, each dataset adds 10% of the data taken from the other 9 datasets to form a new dataset
4. Based on the 2nd method, each dataset adds 50% of the data taken from the other 9 datasets to form a new dataset 
5. Based on the 3rd method, each dataset adds some error label data to form a new dataset

> Usage
Environment: python3.6
```
pip3 install -r requirements.txt
```
step1:  `source ~/venv/bin/activate`

step2: `./splitDataset.sh`

> Parameters

```
split-mode: requirements for daidataset;
```
We will generate 10 custom datasets.
On cifar10 files, every npy file consists of python's List.

```
npy file: [[imgs, label], [imgs, label]...., [imgs, label]]
```
We will use readnpy method to read npy file

```
def readnpy(path):
    # npy file: [[imgs, label], [imgs, label]...., [imgs, label]]
    np_array = np.load(path)
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
readnpy("./cifar10/splitByLabels/splitByLabels_0.npy")
```