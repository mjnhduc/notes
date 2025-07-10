import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Config
EPOCHS = 5
BATCH_SIZE = 32
LR = 0.001

# Dataset
transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True
)

# Model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        return self.fc(x.view(-1, 28*28))

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    # Refactored: Move training loop to a function and call it here
    def train(model, train_loader, criterion, optimizer, epochs):
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Call the refactored train function
    train(model, train_loader, criterion, optimizer, EPOCHS)
