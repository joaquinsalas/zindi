import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load the data
hdf5_file_train = "../data/train_data.h5"
with h5py.File(hdf5_file_train, 'r') as hdf:
    X_train = np.array(hdf['images'])  # (1100000, 16, 16, 6)
    y_train = np.array(hdf['labels'])  # Assuming there's a 'labels' dataset with shape (1100000,)

# Clip negative values to zero
X_train = np.clip(X_train, 0, None)

# Split data into training, validation, and test sets (50%, 20%, 30%)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2857, random_state=42)  # 20% of original

# Flatten the images for the fully connected network: (1100000, 16, 16, 6) -> (1100000, 16*16*6)
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Standardize the data using the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save the standardization parameters
scaler_filename = '../data/standardization_params.npy'
np.save(scaler_filename, scaler.mean_)

# Count the number of 0's and 1's in y_train to calculate the class weights
num_zeros = np.sum(y_train == 0)
num_ones = np.sum(y_train == 1)

# Calculate the weight for the minority class (class 1)
class_weight = num_zeros / num_ones

# Save the class weight
with open('../data/class_weight.txt', 'w') as f:
    f.write(f"Class weight (minority class 1): {class_weight}\n")

# Define the neural network model class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(NeuralNet, self).__init__()
        layers = []
        current_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size

        # Change final layer to output a single value for binary classification
        layers.append(nn.Linear(current_size, 1))  # Output 1 value per sample
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x.squeeze()  # Flatten to match the shape of the target (batch_size,)

# Define a function to train the model
def train_model(model, criterion, optimizer, num_epochs, patience):
    best_loss = np.inf
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32))  # Removed .view(-1, 1)
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.tensor(X_val, dtype=torch.float32))
            val_loss = criterion(val_outputs, torch.tensor(y_val, dtype=torch.float32))  # Removed .view(-1, 1)

        # Save losses
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())

        # Early stopping
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            patience_counter = 0
            # Save the best model
            best_model = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model)
    return model, history

# Define the loss function using the calculated class weight
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight]))

# Hyperparameter search space
layer_sizes = [1, 2, 3]
neuron_counts = np.arange(100, 1101, 100)

best_model = None
best_hyperparams = None
best_auc = 0

# Hyperparameter search
for num_layers in layer_sizes:
    for num_neurons in neuron_counts:
        print(f"Training model with {num_layers} layers and {num_neurons} neurons per layer...")
        hidden_sizes = [num_neurons] * num_layers
        model = NeuralNet(X_train.shape[1], hidden_sizes)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        trained_model, history = train_model(model, criterion, optimizer, num_epochs=1000, patience=10)

        # Validate on the test data
        model.eval()
        with torch.no_grad():
            test_outputs = model(torch.tensor(X_test, dtype=torch.float32))
            test_outputs = torch.sigmoid(test_outputs).numpy()  # Sigmoid activation for binary classification
            fpr, tpr, _ = roc_curve(y_test, test_outputs)
            test_auc = auc(fpr, tpr)

        # Check if this is the best model
        if test_auc > best_auc:
            best_auc = test_auc
            best_model = trained_model
            best_hyperparams = (num_layers, num_neurons)
            print(f"New best model found with AUC: {best_auc}")

# Save the best model and hyperparameters
torch.save(best_model.state_dict(), "../data/best_model_nn.pth")
with open('../data/best_hyperparameters_fc_nn.txt', 'w') as f:
    f.write(f"Best model: {best_hyperparams}, AUC: {best_auc}, Class Weight: {class_weight}\n")

# Plotting the precision-recall and ROC curves on test data
test_probs = test_outputs

# Precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, test_probs)

# Save precision-recall values to CSV
pr_data = pd.DataFrame({'Precision': precision, 'Recall': recall})
pr_data.to_csv('../data/precision_recall_values_fc_nn.csv', index=False)

# Save FPR-TPR values to CSV
roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
roc_data.to_csv('../data/fpr_tpr_values_fc_nn.csv', index=False)

print("Precision-recall and FPR-TPR values saved to CSV files.")

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='white')
plt.title('Precision-Recall Curve', fontsize=16, color='white')
plt.xlabel('Recall', fontsize=14, color='white')
plt.ylabel('Precision', fontsize=14, color='white')
plt.gca().set_facecolor('black')
plt.gca().tick_params(colors='white')
plt.savefig('../figures/precision_recall_curve_fn_nn.png', facecolor='black', dpi=300)
plt.close()

# ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='white')
plt.plot([0, 1], [0, 1], '-', color='white')
plt.title('ROC Curve', fontsize=16, color='white')
plt.xlabel('False Positive Rate', fontsize=14, color='white')
plt.ylabel('True Positive Rate', fontsize=14, color='white')
plt.gca().set_facecolor('black')
plt.gca().tick_params(colors='white')
plt.savefig('../figures/roc_curve_fc_nn.png', facecolor='black', dpi=300)
plt.close()

# Save the training history to a CSV file
history_df = pd.DataFrame(history)
history_df.to_csv('../data/training_history_fc_nn.csv', index=False)

print("Model training and evaluation complete. Best model saved.")



