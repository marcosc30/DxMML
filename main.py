import torch
import os
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import DxMML
from typing import Optional
from tqdm import tqdm
from visualize import visualize_results

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters


    # Make each of the 4 models

    # Dataset

    # Dataloader

    # Loss and optimizers for each model

    # Lists to store training and test losses
    train_losses = []
    test_losses = []
    model_names = []

    ## For each model: Instantiate, define optimizer, train, test loss

    models = {
    }

    for model_name, model_info in models.items():
      model_path = model_info['model_path']
      model = model_info['model']
      model_optimizer = model_info['optimizer']
      index = model_info['index']

      if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
      else:
        train_losses.append(train(model, train_data_loader, model_optimizer, criterion, device, num_epochs))
        torch.save(model.state_dict(), model_path)
      test_losses[index] = test(model, test_data_loader, criterion, device)


    # CNNLSTM = WeatherForecasterCNNLSTM(num_features, hidden_size_cnnlstm, num_layers, output_size, kernel_size, dropout).to(device)
    # cnnlstm_optimizer = Adam(CNNLSTM.parameters(), lr=learning_rate)
    # train_losses.append(train(CNNLSTM, train_data_loader, cnnlstm_optimizer, criterion, device, num_epochs))
    # test_losses[0] = test(CNNLSTM, test_data_loader, criterion, device)

    # CNNTransformer = WeatherForecasterCNNTransformer(num_features, hidden_size_cnntransformer, num_layers, output_size, kernel_size, dropout).to(device)
    # cnntransformer_optimizer = Adam(CNNTransformer.parameters(), lr=learning_rate)
    # train_losses.append(train(CNNTransformer, train_data_loader, cnntransformer_optimizer, criterion, device, num_epochs))
    # # test_losses[1] = test(CNNTransformer, test_data_loader, criterion, device)

    # TCN = TemporalConvNet2D(12, num_features, kernel_size, dropout).to(device)
    # tcn_optimizer = Adam(TCN.parameters(), lr=learning_rate)
    # train_losses.append(train(TCN, train_data_loader, tcn_optimizer, criterion, device, num_epochs))
    # test_losses[2] = test(TCN, test_data_loader, criterion, device)

    # CLSTM = ConvLSTM(num_features, 120, kernel_size, num_layers).to(device)
    # clstm_optimizer = Adam(CLSTM.parameters(), lr=learning_rate)
    # train_losses.append(train(CLSTM, train_data_loader, clstm_optimizer, criterion, device, num_epochs))
    # test_losses[3] = test(CLSTM, test_data_loader, criterion, device)

    # Visualize the results
    visualize_results(train_losses, test_losses, model_names, num_epochs)


def train(model, dataloader, optimizer, criterion, device, num_epochs):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # progress bar to see the loss as the model trains
            progress_bar.set_postfix({'loss': f'{running_loss / len(dataloader):.4f}'})
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")
        losses.append(running_loss/len(dataloader))
    return losses

def test(model, dataloader, criterion, device):
    model.eval()

    print('Test Accuracy: ', accuracy)
    return running_loss / len(dataloader)