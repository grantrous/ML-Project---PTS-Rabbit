import torch
import matplotlib.pyplot as plt

def save_model(model, filepath):
    """Saves the model weights to the specified filepath."""
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    """Loads the model weights from the specified filepath."""
    model.load_state_dict(torch.load(filepath))
    model.eval()

def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plots training and validation loss and accuracy over epochs."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def calculate_accuracy(predictions, labels):
    """Calculates the accuracy of predictions against true labels."""
    _, predicted_classes = torch.max(predictions, 1)
    correct_predictions = (predicted_classes == labels).sum().item()
    accuracy = correct_predictions / labels.size(0)
    return accuracy

#This function visualizes the pressure data and rabbit position over time.
def visualize_data(df1):
    '''Visualizes pressure data and rabbit position over time.'''
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    fig, ax1 = plt.subplots(figsize=(16, 8))
    pressure_cols = [col for col in df1.columns if col.startswith('Pressure_')]
    colors = plt.cm.Accent(np.linspace(0, 1, len(pressure_cols)))  

    # Plot all pressure columns with unique colors
    for idx, col in enumerate(pressure_cols):
        ax1.plot(df1['time'], df1[col], label=col, color=colors[idx])

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Pressure')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

    # Set x-axis major ticks to every Nth value for spacing
    N = max(1, len(df1) // 10)
    ax1.set_xticks(df1['time'][::N])

    fig.autofmt_xdate(rotation=30)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)  # Move legend outside

    # Plot Rabbit Position as scatter on secondary y-axis
    ax2 = ax1.twinx()
    ax2.scatter(df1['time'], df1['RabbitPosition'], color='tab:orange', label='Rabbit Position', alpha=0.7)
    ax2.set_ylabel('Rabbit Position', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('All Pressures and Rabbit Position Over Time')
    fig.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend
    plt.show()


import seaborn as sns
def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=0, ha='right')
  plt.ylabel('True Position')
  plt.xlabel('Predicted Position')

