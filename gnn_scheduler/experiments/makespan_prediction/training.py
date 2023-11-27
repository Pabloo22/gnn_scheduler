import torch
from torch.optim import Adam
from torch.nn import MSELoss
import wandb

from gnn_scheduler.experiments.makespan_prediction import (
    create_data_loader, GCNRegression
)

BATCH_SIZE = 16
LEARNING_RATE = 0.01
EPOCHS = 100


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        # Move data to GPU
        data = data.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    wandb.init(project="jssp-makespan-prediction")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    loader, num_node_features = create_data_loader(batch_size=BATCH_SIZE)
    model = GCNRegression(num_node_features).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = MSELoss()
    
    wandb.config = {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
    }
    # Optionally, watch the model
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Training the model
    for epoch in range(EPOCHS):
        loss = train(model, loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}')

        # Log training metrics
        wandb.log({"epoch": epoch, "loss": loss})

    wandb.finish()

if __name__ == "__main__":
    main()
