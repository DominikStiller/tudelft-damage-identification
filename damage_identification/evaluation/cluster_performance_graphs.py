import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# from cluster_performace import collate_metrics

def graph_metrics(directory):
    dirs = [os.path.join(directory, name) for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    for d in dirs:
        print(d)
        #metrics = collate_metrics(data, d)


print(graph_metrics("data"))