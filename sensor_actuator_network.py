import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# ------------------------------------------------------------------------------
# 10x Improved Codebase for a Sensor-Augmentation Network with recommended fixes:
# 1) Non-deterministic dataset splitting addressed: now uses a seeded generator.
# 2) Removed duplicate seeding inside the dataset.
# 3) Single-device training remains, but we add a note on multi-GPU extension.
# 4) Everything else remains as previously improved.
# ------------------------------------------------------------------------------

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResidualBlock(nn.Module):
    """Residual block with two linear layers and a skip connection."""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += identity
        out = self.relu(out)
        return out


class SensorAugmentor(nn.Module):
    """
    SensorAugmentor:
    1) Learns to reconstruct high-quality (HQ) signals from lower-quality (LQ) signals.
    2) Provides a refined latent representation for actuator control.
    3) Uses a residual architecture for more stable training.
    """
    def __init__(self, sensor_dim=32, hidden_dim=64, output_dim=16, num_resblocks=2):
        super(SensorAugmentor, self).__init__()
        
        # Shared base encoder for sensor signals.
        self.encoder = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.ReLU(),
            *[ResidualBlock(hidden_dim) for _ in range(num_resblocks)]
        )

        # Reconstruction head: from latent -> HQ sensor space.
        self.hq_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sensor_dim)
        )

        # Actuator regression head: from latent -> actuator command.
        self.actuator_head = nn.Linear(hidden_dim, output_dim)

        # Additional residual block on top of hidden representation.
        self.post_encoding_resblock = ResidualBlock(hidden_dim)

    def forward(self, x_lq, x_hq=None):
        # Encode LQ sensor
        encoded_lq = self.encoder(x_lq)
        encoded_lq = self.post_encoding_resblock(encoded_lq)

        # Reconstruct HQ sensor from LQ
        reconstructed_hq = self.hq_reconstructor(encoded_lq)
        # Predict actuator command
        act_command = self.actuator_head(encoded_lq)

        if x_hq is not None:
            # Encode HQ sensor for teacher-student alignment
            encoded_hq = self.encoder(x_hq)
            encoded_hq = self.post_encoding_resblock(encoded_hq)
            return reconstructed_hq, act_command, encoded_lq, encoded_hq
        else:
            return reconstructed_hq, act_command, encoded_lq, None


# Synthetic Dataset
# ------------------------------------------------------------------------------

class SyntheticSensorDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset generating pairs of (LQ, HQ) sensor data and a ground truth actuator command.
    """
    def __init__(self, num_samples=1000, sensor_dim=32, noise_factor=0.3):
        super().__init__()

        # Generate random HQ data (do not re-seed here to avoid overriding global seed)
        x_hq = torch.randn(num_samples, sensor_dim)

        # Generate LQ data by adding noise
        x_lq = x_hq + noise_factor * torch.randn(num_samples, sensor_dim)

        # Normalize data for stability
        self.mean_hq = x_hq.mean(dim=0, keepdim=True)
        self.std_hq = x_hq.std(dim=0, keepdim=True) + 1e-6
        self.mean_lq = x_lq.mean(dim=0, keepdim=True)
        self.std_lq = x_lq.std(dim=0, keepdim=True) + 1e-6

        x_hq_norm = (x_hq - self.mean_hq) / self.std_hq
        x_lq_norm = (x_lq - self.mean_lq) / self.std_lq

        # Example actuator command: sum of first few dims in HQ sensor + some noise.
        y_cmd = torch.sum(x_hq_norm[:, :5], dim=1, keepdim=True)

        self.x_lq = x_lq_norm
        self.x_hq = x_hq_norm
        self.y_cmd = y_cmd

    def __len__(self):
        return self.x_lq.size(0)

    def __getitem__(self, idx):
        return self.x_lq[idx], self.x_hq[idx], self.y_cmd[idx]


# Training Routine
# ------------------------------------------------------------------------------

class EarlyStopper:
    """
    Early stopping to stop training when validation loss doesn't improve after a certain patience.
    """
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def check(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_model(model, train_loader, val_loader=None, epochs=30, lr=1e-3, device="cpu"):
    """
    Train the model with optional validation.
    """
    criterion_reconstruction = nn.MSELoss()
    criterion_actuation = nn.MSELoss()  # For regression

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    early_stopper = EarlyStopper(patience=6, min_delta=1e-4)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x_lq, x_hq, y_cmd in train_loader:
            x_lq = x_lq.to(device)
            x_hq = x_hq.to(device)
            y_cmd = y_cmd.to(device)

            # Forward pass
            reconstructed_hq, act_command, encoded_lq, encoded_hq = model(x_lq, x_hq)

            # Reconstruction loss
            loss_recon = criterion_reconstruction(reconstructed_hq, x_hq)

            # Encoding alignment loss (teacher-student)
            if encoded_hq is not None:
                loss_encoding = criterion_reconstruction(encoded_lq, encoded_hq)
            else:
                loss_encoding = 0

            # Actuator loss
            loss_act = criterion_actuation(act_command, y_cmd)

            # Weighted combined loss
            loss = loss_recon + 0.1 * loss_encoding + loss_act

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_lq.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation step if provided
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for val_x_lq, val_x_hq, val_y_cmd in val_loader:
                    val_x_lq = val_x_lq.to(device)
                    val_x_hq = val_x_hq.to(device)
                    val_y_cmd = val_y_cmd.to(device)

                    pred_hq, pred_act, enc_lq, enc_hq = model(val_x_lq, val_x_hq)

                    val_loss_recon = criterion_reconstruction(pred_hq, val_x_hq)
                    val_loss_encoding = criterion_reconstruction(enc_lq, enc_hq) if enc_hq is not None else 0
                    val_loss_act = criterion_actuation(pred_act, val_y_cmd)

                    val_loss_batch = val_loss_recon + 0.1 * val_loss_encoding + val_loss_act
                    val_running_loss += val_loss_batch.item() * val_x_lq.size(0)

            val_loss = val_running_loss / len(val_loader.dataset)
            scheduler.step(val_loss)

            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            early_stopper.check(val_loss)
            if early_stopper.should_stop:
                print("Early stopping triggered.")
                break
        else:
            # If no val loader, just use training loss for scheduling.
            scheduler.step(epoch_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")


# Usage Demonstration
# ------------------------------------------------------------------------------

def main():
    set_seed(42)

    # Basic parameters
    sensor_dim = 32
    hidden_dim = 64
    output_dim = 1
    batch_size = 32
    epochs = 20
    lr = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare dataset
    dataset = SyntheticSensorDataset(num_samples=2000, sensor_dim=sensor_dim, noise_factor=0.3)

    # Split into train/val with a seeded generator for reproducibility
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    g = torch.Generator()
    g.manual_seed(42)  # same seed used for dataset splitting

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = SensorAugmentor(sensor_dim=sensor_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_resblocks=2)

    # Train model
    train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

    print("Training complete.")

    # Example inference with new LQ data:
    model.eval()
    test_lq = torch.randn(1, sensor_dim).to(device)
    # Normalize test_lq using dataset stats
    test_lq_norm = (test_lq - dataset.mean_lq.to(device)) / dataset.std_lq.to(device)

    reconstructed_hq, act_command, encoded_lq, _ = model(test_lq_norm)
    print("LQ input:", test_lq_norm)
    print("Reconstructed HQ:", reconstructed_hq)
    print("Actuator command:", act_command)

    # NOTE: For multi-GPU training, you can wrap the model:
    # model = nn.DataParallel(model)
    # Then proceed with the same training loop.


if __name__ == "__main__":
    main() 