import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader




def test_model(data_loader: DataLoader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()
    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")