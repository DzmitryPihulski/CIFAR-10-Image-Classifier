import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils.Dataset import CustomImageDataset
from utils.models.ConvNet_1_model import ConvNet_1
from utils.models.ConvNet_2_model import ConvNet_2
from utils.models.ResNet18_model import ResNet18
from utils.utils import set_seed, transform_pipeline

# Function to train a model
def train_model(model, model_name: str):
    """
    Trains the model on CIFAR-10, saves the best model, and plots losses.

    :param model: Model to train (e.g., ResNet18).
    :param model_name: Name used for saving model and results.
    """

    set_seed(42) # Set random seed for reproducibility
    print(f"Training {model_name}")

    # Initialize model, loss function, and optimizer
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())  # Set an initial learning rate

    epochs = np.arange(1, 51)
    losses_train = list()
    losses_val = list()

    # Early stopping parameters
    patience = 10
    min_delta = 0.01
    patience_counter = 0
    best_val_loss = float('inf')

    # Training process
    current_time = time.time()
    for epoch in epochs:
        epoch_loss_train = 0
        running_loss = 0
        model.train()  # Set model to training mode
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss_train += loss.item()

            # Log progress every 500 iterations
            if i % 500 == 499:
                print(f"Current time of execution: {round(time.time() - current_time, 0)} s, epoch: {epoch}/{epochs[-1]}, minibatch: {i + 1:5d}/{len(train_loader)}, running loss: {running_loss / 500:.3f}")
                running_loss = 0

        # Validation phase
        model.eval()  # Set model to evaluation mode
        epoch_loss_val = 0
        with torch.no_grad():
            for data in validation_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_loss_val += loss.item()

        avg_train_loss = epoch_loss_train / len(train_loader)
        avg_val_loss = epoch_loss_val / len(validation_loader)
        losses_train.append(avg_train_loss)
        losses_val.append(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': losses_train,
                'val_loss': losses_val
            }, f'src/data/{model_name}_best_performance.pth')
        else:
            patience_counter += 1
        
        # Trigger early stopping if no improvement
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': losses_train,
        'val_loss': losses_val
    }, f'src/data/{model_name}_full_training.pth')

    print('Finished Training')

    # Plot training and validation losses
    plt.figure(figsize=(10,7))
    marker_on = [losses_val.index(min(losses_val))]
    plt.plot(np.arange(1, len(losses_train)+1), losses_train, color='r', label="Train loss")
    plt.plot(np.arange(1, len(losses_val)+1), losses_val, '-gD', label="Test loss", markevery=marker_on)
    bbox = dict(boxstyle ="round", fc ="0.8")
    plt.annotate(text=f"Min loss after {losses_val.index(min(losses_val))+1} epohs", xy=(losses_val.index(min(losses_val))+1, min(losses_val)), xytext =(losses_val.index(min(losses_val))+1, min(losses_val)+0.05),  
                    arrowprops = dict(facecolor ='green', 
                                    shrink = 0.2),bbox=bbox)
    plt.grid()
    plt.title(f"Train and Val loss after {len(losses_train)} epochs")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropyLoss")
    plt.legend(loc='upper left')
    plt.savefig(f"src/data/images/{model_name}.png")

    # Calculate test accuracy
    total_correct = 0
    total_samples = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Update the running total of correct predictions and samples
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    # Calculate the accuracy for this epoch
    accuracy = 100 * total_correct / total_samples
    print(f'Accuracy = {accuracy:.2f}%')

#--------------------------------------------------------------------

# Set device to GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize datasets
train_set = CustomImageDataset()
validation_set = CustomImageDataset()
test_set = CustomImageDataset()

# Load batches for training
train_set.load_batch('src/data/cifar-10-batches-py/data_batch_1')
train_set.load_batch_transformed('src/data/cifar-10-batches-py/data_batch_1', transform_pipeline)
train_set.load_batch('src/data/cifar-10-batches-py/data_batch_2')
train_set.load_batch_transformed('src/data/cifar-10-batches-py/data_batch_2', transform_pipeline)
train_set.load_batch('src/data/cifar-10-batches-py/data_batch_3')
train_set.load_batch_transformed('src/data/cifar-10-batches-py/data_batch_3', transform_pipeline)
train_set.load_batch('src/data/cifar-10-batches-py/data_batch_4')
train_set.load_batch_transformed('src/data/cifar-10-batches-py/data_batch_4', transform_pipeline)

# Load validation and test data
validation_set.load_batch('src/data/cifar-10-batches-py/data_batch_5')
test_set.load_batch('src/data/cifar-10-batches-py/test_batch')

# DataLoader for batching
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                         shuffle=True, pin_memory=True)

validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32,
                                         shuffle=True, pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,
                                         shuffle=True, pin_memory=True)

print("Data loading finished!")

# Train models
train_model(ConvNet_1, 'ConvNet_1')
train_model(ConvNet_2, 'ConvNet_2')
train_model(ResNet18, 'ResNet18')