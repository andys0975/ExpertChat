import json
from glob import glob

import torch
import numpy as np
import pandas as pd

class Data:
    def __init__(self):
        self.expert_intro = None
        self.expert_intro_vectors = None
        self.publication = None
        self.publication_vectors = None

        with open("reference/expert_intro.json") as f: 
            self.expert_intro = json.load(f)["expert"]

        self.publication = {path.split("/")[-1].split(".")[0]: pd.read_csv(path) for path in glob("reference/*.csv")}
        self.publication_vectors = dict()
        for path in glob("vectors/*.npy"):
            with open(path, 'rb') as f: 
                if "expert_intro" in path: 
                    self.expert_intro_vectors = torch.tensor(np.load(f), device="mps")
                else: 
                    self.publication_vectors[path.replace("_vectors","").split("/")[-1].split(".")[0]] = torch.tensor(np.load(f), device="mps")
