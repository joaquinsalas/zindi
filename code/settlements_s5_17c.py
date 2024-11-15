import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm.auto import tqdm
import argparse
from s5 import S5Block  # Import S5Block for the S5 model


# Parser setup
parser = argparse.ArgumentParser(description='Settlement Identification using S5')
# Optimizer settings
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for dataloader')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
args = parser.parse_args()


# Dataset Class with Flattening
class H5Dataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file

        with h5py.File(h5_file, 'r') as hdf:
            self.images = np.clip(np.array(hdf['images']), 0, None) * 0.00005
            self.labels = np.array(hdf['labels'])
        self.features = self.aggregate_features(self.images)


    def aggregate_features(self, images):
        spectral_indices = self.calculate_spectral_indices(images)
        features = np.concatenate((images, spectral_indices), axis=-1)
        features = np.nan_to_num(features, nan=0.0, posinf=1e30, neginf=-1e30)
        return features

    def calculate_spectral_indices(self, data):
        blue, green, red, nir, swir1, swir2 = np.split(data, 6, axis=-1)
        epsilon = 1e-8
        ndvi = (nir - red) / (nir + red + epsilon)
        ndvi = np.clip(ndvi, -1, 1)
        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1 + epsilon))
        evi = np.clip(evi, -1,1)
        ndwi = (green - nir) / (green + nir + epsilon)
        ndwi = np.clip(ndwi, -1, 1)
        ndbi = (swir1 - nir) / (swir1 + nir + epsilon)
        ndbi = np.clip(ndbi, -1, 1)
        savi = ((nir - red) / (nir + red + 0.5 + epsilon)) * 1.5
        savi = np.clip(savi, -1, 1)
        nbr = (nir - swir2) / (nir + swir2 + epsilon)
        nbr = np.clip(nbr, -1, 1)
        evi2 = 2.5 * ((nir - red) / (nir + 2.4 * red + 1 + epsilon))
        evi2 = np.clip(evi2, -1, 1)
        msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2
        msavi = np.clip(msavi, -1, 1)
        nmdi = (nir - (swir1 - swir2)) / (nir + (swir1 - swir2) + epsilon)
        nmdi = np.clip(nmdi, -1, 1)
        ndi45 = (red - blue) / (red + blue + epsilon)
        ndi45 = np.clip(ndi45, -1, 1)
        si = (blue + green + red) / 3
        si = np.clip(si, 0, 1)
        return np.concatenate((ndvi, evi, ndwi, ndbi, savi,
                               nbr, evi2, msavi,
                               nmdi, ndi45, si), axis=-1)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        features = torch.tensor(features, dtype=torch.float32).view(256,17)
        label = torch.tensor(label, dtype=torch.float32)
        return features, label

# Define the S5 model for a flattened input
class S5Model(nn.Module):
    def __init__(self, d_input, d_output=1, d_model=1024, n_layers=4, dropout=0.2, prenorm=False):
        super(S5Model, self).__init__()
        self.prenorm = prenorm
        self.encoder = nn.Linear(d_input, d_model)
        self.s5_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s5_layers.append(S5Block(dim=d_model, state_dim=d_model, bidir=False))


            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        x = self.encoder(x)  # (B, 256, 17) -> (B, d_model)

        for layer, norm, dropout in zip(self.s5_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                # Apply normalization only on the intended dimension
                z = norm(z)  # Remove any additional transposition
            z  = layer(z)  # Assuming layer expects input with shape (B, d_model, Seq_len)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x)  # Keep it without transposing if norm applies on (B, d_model, Seq_len)

        x = x.mean(dim=1)  # Average over the sequence dimension, keeping (B, d_model)
        x = self.decoder(x)  # Final output (B, d_output)
        return x

# Function to load model, or load existing best model weights if available
def load_model(device):
    model = S5Model(d_input=17, d_output=1, d_model=1024, n_layers=2, dropout=0.1, prenorm=True).to(device)
    if os.path.exists(model_save_path):
        print("Loading existing best model weights...")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    return model


# Training function
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return running_loss / len(train_loader)

# Evaluation function
def evaluate_model(model, val_loader, device):
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    auc_score = roc_auc_score(np.array(all_labels), np.array(all_outputs).flatten())
    return auc_score

# Function to set up the optimizer and scheduler
def setup_optimizer(model, lr, weight_decay, epochs):
    all_parameters = list(model.parameters())
    params = [p for p in all_parameters if not hasattr(p, "_optim")]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, scheduler

# Main function

def main():

    batch_size = 512
    epochs = 1000
    patience = 0

    dataset = H5Dataset(h5_file)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # Separate majority and minority class indices
    train_labels = [train_set[i][1].item() for i in range(len(train_set))]
    minority_class_indices = np.where(np.array(train_labels) == 1)[0]
    majority_class_indices = np.where(np.array(train_labels) == 0)[0]

    print(f'Minority class size: {len(minority_class_indices)}, Majority class size: {len(majority_class_indices)}')

    # Check if majority class is large enough
    assert len(majority_class_indices) >= len(
        minority_class_indices), "Majority class has fewer elements than expected."

    overall_best_val_auc = 0.0

    majority_class_chunks = np.array_split(majority_class_indices,
                                           len(majority_class_indices) // len(minority_class_indices))

    for chunk_idx, chunk in enumerate(majority_class_chunks):
        print(f"Training classifier {chunk_idx + 1}/{len(majority_class_chunks)}")

        # Create balanced dataset for this classifier
        balanced_indices = np.concatenate((minority_class_indices, chunk))
        balanced_train_set = Subset(train_set, balanced_indices)


        train_loader = DataLoader(balanced_train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Load model with existing best weights if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(device)
        model = model.to(device)

        optimizer, scheduler = setup_optimizer(model, lr=learning_rate, weight_decay=0.01, epochs=epochs)
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = 0.0
        epochs_no_improve = 0
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            train_loss = train_model(model, train_loader, optimizer, criterion, device)
            val_auc = evaluate_model(model, val_loader, device)

            print(f'Validation ROC AUC: {val_auc:.4f}')
            if val_auc > best_val_auc:
                best_val_auc = val_auc

                if val_auc > overall_best_val_auc:
                    overall_best_val_auc = val_auc
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Best model saved with AUC: {overall_best_val_auc:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                print(f'Early stopping after {patience} epochs without improvement.')
                break

            scheduler.step()


model_path = '../models/'
model_save_path = model_path + 's5_model_17c.pth'  # Path to save and load the best model
h5_file = '../data/train_data.h5'
learning_rate = 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    main()
