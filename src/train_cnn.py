import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from dataset import TextureDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TextureDataset("data/images")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        pred = model(imgs)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch", epoch, "Loss:", total_loss)

torch.save(model.state_dict(), "models/cnn.pth")