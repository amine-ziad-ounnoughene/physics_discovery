import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tools import *
def prepare_data_loaders(dataset, n_equation, batch_size=50, equation_size=30):
    x_train, y_train = data_prep(dataset, "train", n_equation, equation_size)
    labels_train = standardize_tensor(torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    features_train = standardize_tensor(torch.tensor(x_train, dtype=torch.float32), labels_train)
    train_dataset = TensorDataset(features_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    x_test, y_test = data_prep(dataset, "test", n_equation, equation_size)
    labels_test = standardize_tensor(torch.tensor(y_test, dtype=torch.float32), labels_train)
    features_test = standardize_tensor(torch.tensor(x_test, dtype=torch.float32), labels_test)
    test_dataset = TensorDataset(features_test, labels_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, features_train, labels_train, features_test, labels_test

def train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, num_epochs):
    hist_train_loss = []
    hist_test_loss = []
    for epoch in range(num_epochs):
        losses = []
        w_losses = []
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            output, formula = model(batch_features)
            real_grad_ = real_grad(batch_features)
            fake_grad_ = compute_gradient_last_to_first(model, batch_features)
            grad_loss = criterion(real_grad_, fake_grad_)
            loss = criterion(output.squeeze(1).requires_grad_(), batch_labels)
            w_loss = loss + 0 * grad_loss
            losses.append(loss)
            w_losses.append(w_loss)
            w_loss.backward()
            optimizer.step()
            scheduler.step()
        
        with torch.no_grad():
            test_losses = []
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                output, formula = model(batch_features)
                test_loss = criterion(output.squeeze(1).requires_grad_(), batch_labels)
                test_losses.append(test_loss.item())
            avg_test_loss = sum(test_losses) / len(test_losses)

        avg_train_loss = sum(losses) / len(losses)
        hist_train_loss.append(avg_train_loss)
        hist_test_loss.append(avg_test_loss)
        print(f"EPOCH[{epoch}] test loss: {avg_test_loss:.6f}, train loss: {avg_train_loss:.6f}, weighted_loss: {sum(w_losses) / len(w_losses):.6f}")

    return hist_train_loss, hist_test_loss
