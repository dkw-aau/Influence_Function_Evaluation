from image_explainer import *
from utils2 import *
import numpy as np
import torch
import torchvision
from modules import *
from tqdm import tqdm


embeds = load_inception_embeds()
X_train = torch.tensor(embeds["X_train"])
Y_train = torch.tensor(embeds["Y_train"])

X_test =  torch.cat((torch.tensor(embeds["X_test"]), torch.tensor(embeds["X_train"])), dim=0)
Y_test = torch.cat((torch.tensor(embeds["Y_test"]), torch.tensor(embeds["Y_train"])), dim=0)

train_set = data.TensorDataset(X_train, Y_train)
test_set = data.TensorDataset(X_test, Y_test)
clf = fit_model(X_train, Y_train)

module = LiSSAInfluenceModule(
    model=clf,
    objective=BinClassObjective(),
    train_loader=data.DataLoader(train_set, batch_size=32),
    test_loader=data.DataLoader(test_set, batch_size=32),
    device=DEVICE,
    damp=0.001,
    repeat= 1,
    depth=1800,
    scale= 10,
)
train_idxs = list(range(X_train.shape[0]))

a= []
for i in tqdm(range(len(Y_test))):
    if module.influences(train_idxs=train_idxs, test_idxs=[i]).sum()==0:
        a.append(i)

with open("weirdos.txt", "w") as f:
    for s in a:
        f.write(str(s) +"\n")
print(a)