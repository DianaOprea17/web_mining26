# Exercise 2 (15 minutes): Building an MLP Using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Create dataset: 2D points, label = 1 if y > x else 0
torch.manual_seed(42)

X = torch.rand(200, 2)
y = (X[:, 1] > X[:, 0]).float().unsqueeze(1)  # shape (200, 1)

# 2. Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = MLP()

# 3. Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
losses = []

epochs = 100

for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 5. Evaluation
with torch.no_grad():
    predictions = model(X)
    predicted_labels = (predictions > 0.5).float()

    accuracy = (predicted_labels == y).float().mean()
    print(f"\nTraining Accuracy: {accuracy.item():.4f}")

# 6. Plot loss over time
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True)
plt.show()