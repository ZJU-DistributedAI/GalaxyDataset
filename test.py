import numpy as np
# array = [[1, 2], 3]
#
# np.save("./test.npy", array)
#
# print(np.load("./test.npy", allow_pickle=True))

import yaml
import os
f = open("./config.yaml")
y = yaml.load(f)
print(y)
print(y["split_mode"])


# def readYaml(path, args):
#     if not os.path.exists(path):
#         return args
#     f = open(path)
#     config = yaml.load(f)
#     args.node_num = int(config[0]["node_num"])
#     args.isaverage_dataset_size = config[0]["isaverage_dataset_size"]
#     args.dataset_size_list = config[0]["dataset_size_list"]
#     args.split_mode = int(config[0]["split_mode"])
#     args.node_label_num = config[0]["node_label_num"]
#     args.isadd_label = config[0]["isadd_label"]
#     args.add_label_rate = float(config[0]["add_label_rate"])
#     args.isadd_error = config[0]["isadd_error"]
#     args.add_error_rate = float(config[0]["add_error_rate"])
#     return args
#
# readYaml("./config.yaml")