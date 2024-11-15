import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Updated to use GPU 2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from s5 import S5Block  # Import S5Block for the S5 model




# Dataset Class with 17-channel data
class H5Dataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(h5_file, 'r') as hdf:
            self.images = np.clip(np.array(hdf['images']), 0, None) * 0.00005
        self.features = self.aggregate_features(self.images)

    def aggregate_features(self, images):
        spectral_indices = self.calculate_spectral_indices(images)
        features = np.concatenate((images, spectral_indices), axis=-1)
        return features

    def calculate_spectral_indices(self, data):
        blue, green, red, nir, swir1, swir2 = np.split(data, 6, axis=-1)
        epsilon = 1e-8
        ndvi = (nir - red) / (nir + red + epsilon)
        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1 + epsilon))
        ndwi = (green - nir) / (green + nir + epsilon)
        ndbi = (swir1 - nir) / (swir1 + nir + epsilon)
        savi = ((nir - red) / (nir + red + 0.5 + epsilon)) * 1.5
        nbr = (nir - swir2) / (nir + swir2 + epsilon)
        evi2 = 2.5 * ((nir - red) / (nir + 2.4 * red + 1 + epsilon))
        msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2
        nmdi = (nir - (swir1 - swir2)) / (nir + (swir1 - swir2) + epsilon)
        ndi45 = (red - blue) / (red + blue + epsilon)
        si = (blue + green + red) / 3
        # Clipping each index to typical ranges
        return np.concatenate([
            np.clip(ndvi, -1, 1), np.clip(evi, -1, 1), np.clip(ndwi, -1, 1),
            np.clip(ndbi, -1, 1), np.clip(savi, -1, 1), np.clip(nbr, -1, 1),
            np.clip(evi2, -1, 1), np.clip(msavi, -1, 1), np.clip(nmdi, -1, 1),
            np.clip(ndi45, -1, 1), np.clip(si, 0, 1)
        ], axis=-1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        features = self.features[idx]

        features = torch.tensor(features, dtype=torch.float32).view(256, 17)  # Reshape to 256 x 17

        return features

# S5 Model Definition
# Define the S5 model for 17-channel input
class S5Model(nn.Module):
    def __init__(self, d_input=17, d_output=1, d_model=1024, n_layers=4, dropout=0.2, prenorm=False):
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
        x = self.encoder(x)
        for layer, norm, dropout in zip(self.s5_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z)
            z = layer(z)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x)
        x = x.mean(dim=1)
        x = self.decoder(x)
        return x

# Generate predictions
def generate_predictions(model, test_loader, device):
    model.eval()
    test_predictions = []
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", leave=False)
        for images in progress_bar:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            test_predictions.extend(probs)
    return np.array(test_predictions).flatten()

# Create submission CSV file
def create_submission(test_predictions):
    id_map = pd.read_csv('../data/id_map.csv')
    if len(id_map) != len(test_predictions):
        raise ValueError(f"Length of id_map ({len(id_map)}) and test predictions ({len(test_predictions)}) do not match.")
    submission_df = pd.DataFrame({
        'ID': id_map['id'],
        'Target': test_predictions
    })
    submission_file = submission_path + '/s5_model_submission.csv'
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission file saved to {submission_file}")

# Load test data from HDF5 file
def load_test_data(hdf5_file_test):
    return H5Dataset(hdf5_file_test)

# Main function
def main():

    batch_size = 512

    # Load the pre-trained model
    model = S5Model(d_input=17, d_output=1, d_model=1024, n_layers=2, dropout=0.1, prenorm=True)
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model = model.to(device)

    # Load test data and create a DataLoader
    test_dataset = load_test_data(h5_file_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Generate predictions on the test data
    test_predictions = generate_predictions(model, test_loader, device)

    # Create the submission CSV file
    create_submission(test_predictions)


h5_file_test = '../data/test_data.h5'
submission_path = '../submissions/'
model_path = '../models/'
model_load_path = model_path + 's5_model_17c.pth'  # Path to save and load the best model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    if not os.path.exists(submission_path):
        os.makedirs(submission_path)
    main()
