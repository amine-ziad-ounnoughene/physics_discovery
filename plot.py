import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Formuler
from tools import *
from model import *
from train import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def plot_gradient_distributions(formuler, features_, device):
    _, formula, probas, ops = formuler.forward_test(features_.to(device))

    for m in range(len(probas)):
        proba = probas[m].detach().cpu().numpy()
        op = [" ".join([str(k) for k in g]) for g in ops[m]]
        for v in range(proba.shape[1]):
            data_np = proba[:, v, :]

            # Calculate median values for each column
            medians = np.median(data_np, axis=0)

            # Sort the columns based on median values
            sorted_indices = np.argsort(medians)[::-1]
            sorted_data = data_np[:, sorted_indices]
            sorted_ops = [op[i] for i in sorted_indices]

            # Create box plots for each sorted column with operation values on x-axis
            fig = go.Figure()
            for col, sorted_op in zip(sorted_data.T, sorted_ops):
                fig.add_trace(go.Box(y=col, name=f"'{sorted_op}'", boxmean=True))

            # Update layout
            fig.update_layout(
                xaxis=dict(title='Operations'),
                yaxis=dict(title='Gradient Values'),
                title='Per Operation Gradient Values Distributions (Sorted)',
                showlegend=False
            )

            # Show the plot
            fig.show()

            # Save the plot (optional)
            # fig.write_image(f"plot/plot{m}-{i}.pdf")

def plot_loss_curves(hist_train_loss, hist_test_loss, start=0):
    hist_train_loss_cpu = [i.item() for i in hist_train_loss]
    hist_test_loss_cpu = [i for i in hist_test_loss]
    epochs = range(1 + start, len(hist_train_loss_cpu) + 1)

    # Plotting the learning curves
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, hist_train_loss_cpu[start:], 'b', label='Training loss')
    plt.plot(epochs, hist_test_loss_cpu[start:], 'r', label='Validation/Test loss')
    plt.title('Training and Validation/Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_grid(features, predictions, labels, mode):
    input_size = features.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=input_size, figsize=(input_size * 5, 5))
    size = 2
    alpha = 0.4

    for i in range(input_size):
        axes[i].scatter(features[:, i], predictions, color="blue", label='Prediction', s=size)
        axes[i].scatter(features[:, i], labels, color="red", label='Actual', s=size, alpha=alpha)
        axes[i].set_title(f'Feature {i}')
        axes[i].set_xlabel(f'Feature {i} Value')
        axes[i].set_ylabel('Output')
        axes[i].set_ylim(0)
        axes[i].legend()
        axes[i].grid(True)

    plt.suptitle(mode)
    plt.tight_layout()
    plt.show()
