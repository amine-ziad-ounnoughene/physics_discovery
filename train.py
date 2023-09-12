import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
import json
import os
import numpy as np
from models import SciNet
from utils import target_loss
from loader import build_dataloader

def generate_data(size, size_, t_max):
    t = np.linspace(0, t_max, size)
    min_fr, max_fr = 0.01, 100
    fr = np.linspace(min_fr, max_fr, size_)
    start_st, end_st = 0.01, 100
    st = np.logspace(np.log10(start_st), np.log10(end_st), size_, endpoint=True)

    def f(t, st, fr):
        return st**2 * fr * (1 - t/st - np.exp(-t/st))

    data = []
    for st_ in st:
        for fr_ in fr:
            example = list(f(t, st_, fr_))
            t_pred = np.random.uniform(0, t_max)
            pred = f(t_pred, st_, fr_)
            example.extend([fr_, st_, t_pred, pred])
            data.append(example)

    columns = [str(i) for i in range(size)]
    columns.extend(["fr", "st", "t_pred", "pred"])
    df = pd.DataFrame(data, columns=columns)
    return df

def train_sci_net(scinet, dataloader, optimizer, scheduler, beta, N_EPOCHS, device):
    hist_error = []
    hist_kl = []
    hist_loss = []

    for epoch in range(N_EPOCHS):
        epoch_error = []
        epoch_kl = []
        epoch_loss = []
        for minibatch in dataloader:
            time_series, fr, st, question, answer = (
                minibatch['time_series'].to(device) / 5,
                minibatch['fr'].to(device) / 5,
                minibatch['st'].to(device) / 5,
                minibatch['question'].to(device) / 5,
                minibatch['answer'].to(device) / 5
            )
            inputs = torch.cat((time_series, question.view(-1, 1)), 1)
            outputs = answer

            optimizer.zero_grad()
            pred = scinet.forward(inputs)
            loss_ = target_loss(pred, outputs)
            kl = beta * scinet.kl_loss
            loss = loss_ + kl
            loss.backward()
            optimizer.step()
            error = torch.mean(torch.sqrt((pred[:, 0] - outputs)**2)).detach().cpu().numpy()
            epoch_error.append(float(error))
            epoch_kl.append(float(kl.data.detach().cpu().numpy()))
            epoch_loss.append(float(loss_.data.detach().cpu().numpy()))

        hist_error.append(np.mean(epoch_error))
        hist_loss.append(np.mean(epoch_loss))
        hist_kl.append(np.mean(epoch_kl))

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: SGD lr %.6f -> %.6f" % (epoch+1, before_lr, after_lr))
        print("Epoch %d -- loss %f, RMS error %f, KL %f" % (epoch+1, hist_loss[-1], hist_error[-1], hist_kl[-1]))

    return hist_error, hist_kl, hist_loss

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sizes = [5, 10, 25, 50, 100, 150, 200, 300]
    N_EPOCHS = 150  
    size_ = 200
    t_max = 5
    data_file = "data.csv"
    log_file = "log.json"

    os.remove(log_file)
    with open(log_file, "w") as f:
        json.dump({"runs": []}, f)

    for size in sizes: 
        df = generate_data(size, size_, t_max)
        df.to_csv(data_file)

        scinet = SciNet(size, 1, 3, 100).to(device)  # Move the model to the GPU
        dataloader = build_dataloader(size=size, batch_size=128)

        SAVE_PATH = f"trained_models/scinet1-{size}epoch{N_EPOCHS}.dat"
        optimizer = optim.Adam(scinet.parameters(), lr=0.001)

        hist_error, hist_kl, hist_loss = train_sci_net(scinet, dataloader, optimizer, lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.009, total_iters=N_EPOCHS), 0.5, N_EPOCHS, device)

        torch.save(scinet.state_dict(), SAVE_PATH)
        print(f"Model saved to {SAVE_PATH}")

        with open(log_file, "r") as json_file:
            loaded_data = json.load(json_file)["runs"]

        loaded_data.append({
            "epochs": N_EPOCHS,
            "size": size,
            "kl": hist_kl,
            "RMSE": hist_error,
            "loss": hist_loss
        })

        with open(log_file, "w") as json_file:
            json.dump({"runs": loaded_data}, json_file)

if __name__ == "__main__":
    main()
