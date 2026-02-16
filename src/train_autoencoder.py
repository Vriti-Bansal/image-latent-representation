import torch
from torch.utils.data import DataLoader
from dataset import TextureDataset
from autoencoder import AutoEncoder
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TextureDataset("../data/images")
loader = DataLoader(dataset,batch_size=32,shuffle=True)

model = AutoEncoder().to(device)
opt = torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(5):
    total=0
    for x,_ in loader:
        x=x.to(device)
        out=model(x)
        loss=loss_fn(out,x)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total+=loss.item()
    print("epoch",epoch,total)

torch.save(model.state_dict(),"../models/ae.pth")