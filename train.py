import torch
import tqdm
from test import test_model


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for i, (X, y) in enumerate(data_loader):
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if str(total_loss) == 'nan':
            print(f"nan on iteration {i}")
            break

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    return model





def train_lstm(model, loss_function, optimizer, train_loader, test_loader, forecast_lead, epochs):
    print("Untrained test\n--------")
    test_model(test_loader, model, loss_function)

    for ix_epoch in tqdm.tqdm(range(epochs)):
        print(f"Epoch {ix_epoch}\n---------")
        model = train_model(train_loader, model, loss_function, optimizer=optimizer)
        test_model(test_loader, model, loss_function)
        print()
    torch.save(model, f"./sr_lstm_model_{forecast_lead//2}h_forcast.pth")
    return model
