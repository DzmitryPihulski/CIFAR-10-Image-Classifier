import matplotlib.pyplot as plt
import torch
import numpy as np
from utils.Dataset import CustomImageDataset
import time
from utils.models import ConvNet_1, ConvNet_2, ResNet18
from utils.utils import set_seed
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

train_set = CustomImageDataset()
validation_set = CustomImageDataset()
test_set = CustomImageDataset()


transform_pipeline = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(degrees=15),
])

train_set.load_batch('data/cifar-10-batches-py/data_batch_1')
train_set.load_batch_transformed('data/cifar-10-batches-py/data_batch_1', transform_pipeline)
train_set.load_batch('data/cifar-10-batches-py/data_batch_2')
train_set.load_batch_transformed('data/cifar-10-batches-py/data_batch_2', transform_pipeline)
train_set.load_batch('data/cifar-10-batches-py/data_batch_3')
train_set.load_batch_transformed('data/cifar-10-batches-py/data_batch_3', transform_pipeline)
train_set.load_batch('data/cifar-10-batches-py/data_batch_4')
train_set.load_batch_transformed('data/cifar-10-batches-py/data_batch_4', transform_pipeline)

validation_set.load_batch('data/cifar-10-batches-py/data_batch_5')

test_set.load_batch('data/cifar-10-batches-py/test_batch')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                         shuffle=True, pin_memory=True)

validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32,
                                         shuffle=True, pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,
                                         shuffle=True, pin_memory=True)

print("Data loading finished!")

set_seed(42)
print("Training of Simple Conv net")
net = ConvNet_1()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters())  # Set an initial learning rate

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
    net.train()  # Set the model to training mode
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss_train += loss.item()
        if i % 500 == 499:
            print(f"Current time of execution: {round(time.time() - current_time, 0)} s, epoch: {epoch}/{epochs[-1]}, minibatch: {i + 1:5d}/{len(train_loader)}, running loss: {running_loss / 500:.3f}")
            running_loss = 0

    # Validation phase
    net.eval()  # Set the model to evaluation mode
    epoch_loss_val = 0
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
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
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': losses_train,
            'val_loss': losses_val
        }, 'data/conv_net_1_best_performance.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# Save the final model
torch.save({
    'epoch': epoch,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': losses_train,
    'val_loss': losses_val
}, 'data/conv_net_1_full_training.pth')

print('Finished Training')

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
plt.savefig("data/Conv_1.png")


total_correct = 0
total_samples = 0

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)

    # Forward pass
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # Update the running total of correct predictions and samples
    total_correct += (predicted == labels).sum().item()
    total_samples += labels.size(0)

# Calculate the accuracy for this epoch
accuracy = 100 * total_correct / total_samples
print(f'Accuracy = {accuracy:.2f}%')

set_seed(42)
print("Training of More complicated Conv net")
net = ConvNet_2()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters())  # Set an initial learning rate

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
    net.train()  # Set the model to training mode
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss_train += loss.item()
        if i % 500 == 499:
            print(f"Current time of execution: {round(time.time() - current_time, 0)} s, epoch: {epoch}/{epochs[-1]}, minibatch: {i + 1:5d}/{len(train_loader)}, running loss: {running_loss / 500:.3f}")
            running_loss = 0

    # Validation phase
    net.eval()  # Set the model to evaluation mode
    epoch_loss_val = 0
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
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
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': losses_train,
            'val_loss': losses_val
        }, 'data/conv_net_2_best_performance.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# Save the final model
torch.save({
    'epoch': epoch,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': losses_train,
    'val_loss': losses_val
}, 'data/conv_net_2_full_training.pth')

print('Finished Training')

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
plt.savefig("data/Conv_2.png")


total_correct = 0
total_samples = 0

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)

    # Forward pass
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # Update the running total of correct predictions and samples
    total_correct += (predicted == labels).sum().item()
    total_samples += labels.size(0)

# Calculate the accuracy for this epoch
accuracy = 100 * total_correct / total_samples
print(f'Accuracy = {accuracy:.2f}%')


set_seed(42)
print("Training of ResNet18")
net = ResNet18()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters())  # Set an initial learning rate

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
    net.train()  # Set the model to training mode
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss_train += loss.item()
        if i % 500 == 499:
            print(f"Current time of execution: {round(time.time() - current_time, 0)} s, epoch: {epoch}/{epochs[-1]}, minibatch: {i + 1:5d}/{len(train_loader)}, running loss: {running_loss / 500:.3f}")
            running_loss = 0

    # Validation phase
    net.eval()  # Set the model to evaluation mode
    epoch_loss_val = 0
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
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
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': losses_train,
            'val_loss': losses_val
        }, 'data/ResNet18_best_performance.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# Save the final model
torch.save({
    'epoch': epoch,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': losses_train,
    'val_loss': losses_val
}, 'data/ResNet18_full_training.pth')

print('Finished Training')

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
plt.savefig("data/ResNet18.png")


total_correct = 0
total_samples = 0

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)

    # Forward pass
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # Update the running total of correct predictions and samples
    total_correct += (predicted == labels).sum().item()
    total_samples += labels.size(0)

# Calculate the accuracy for this epoch
accuracy = 100 * total_correct / total_samples
print(f'Accuracy = {accuracy:.2f}%')
