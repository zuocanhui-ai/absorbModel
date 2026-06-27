# FNN Model - Biochar Uranium Adsorption Data
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Set random seed
torch.manual_seed(42)
np.random.seed(42)


# ========== Define FNN Model ==========
class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, dropout=0.2):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Read CSV data
data = pd.read_csv("REAL_biochar_adsorption_ECs_mapped.csv")

# View data information
print("Data shape:", data.shape)
print("\nColumn names:")
print(data.columns.tolist())
print("\nFirst 5 rows:")
print(data.head())

# Feature columns and target column
feature_columns = [
    'SA (m2/g)', 'Dav (nm)', 'VTot (cm3/g)', 'C (wt%)',
    'O/C', '(O+N)/C', 'pH', 'T (K)', 'C0 (mg/L)', 'SLR (g/L)'
]
target_column = 'Qe (mg/g)'

# Extract features and target variables
X = data[feature_columns].values
y = data[target_column].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

# Split train-test set (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} ({X_train.shape[0] / len(X) * 100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0] / len(X) * 100:.1f}%)")

# Feature standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create data loader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train FNN model
print("\n========== Training FNN Model ==========")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = FNN(input_dim=X_train_scaled.shape[1], hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, dropout=0.2)
model = model.to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
n_epochs = 100
train_losses = []
val_losses = []

print(f"Starting training for {n_epochs} epochs...")

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Calculate average training loss
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loss
    model.eval()
    with torch.no_grad():
        X_test_device = X_test_tensor.to(device)
        y_test_device = y_test_tensor.to(device)
        val_outputs = model(X_test_device)
        val_loss = criterion(val_outputs, y_test_device).item()
        val_losses.append(val_loss)

    # Print every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

print("Model training completed!")

# Prediction
model.eval()
with torch.no_grad():
    X_train_device = X_train_tensor.to(device)
    X_test_device = X_test_tensor.to(device)

    y_train_pred = model(X_train_device).cpu().numpy().flatten()
    y_test_pred = model(X_test_device).cpu().numpy().flatten()

# ========== Training Set Evaluation ==========
print("\n========== Training Set Performance Metrics ==========")
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

print(f"MAE  (Mean Absolute Error):  {mae_train:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse_train:.2f}")
print(f"R²   (R-squared):            {r2_train:.2f}")

# ========== Test Set Evaluation ==========
print("\n========== Test Set Performance Metrics ==========")
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print(f"MAE  (Mean Absolute Error):  {mae_test:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse_test:.2f}")
print(f"R²   (R-squared):            {r2_test:.2f}")

# ========== Comparison Analysis ==========
print("\n========== Training vs Test Set Comparison ==========")
comparison_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Training': [f'{mae_train:.2f}', f'{rmse_train:.2f}', f'{r2_train:.2f}'],
    'Test': [f'{mae_test:.2f}', f'{rmse_test:.2f}', f'{r2_test:.2f}'],
    'Difference': [f'{mae_test - mae_train:.2f}',
                   f'{rmse_test - rmse_train:.2f}',
                   f'{r2_test - r2_train:.2f}']
})
print(comparison_df.to_string(index=False))

# ========== Model Information ==========
print("\n========== FNN Model Information ==========")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Network structure: Input(10) → Hidden1(128) → Hidden2(64) → Hidden3(32) → Output(1)")
print(f"Dropout rate: 0.2 (20%)")
print(f"Activation function: ReLU")

# ========== Model Evaluation Summary ==========
print("\n========== Model Evaluation Summary ==========")
if abs(r2_train - r2_test) < 0.05:
    print("✓ Good generalization, training and test performance are close")
elif r2_train > r2_test + 0.05:
    print("⚠ Possible overfitting, test performance significantly lower than training")
    print("  Suggestion: Increase dropout or reduce network complexity")
else:
    print("! Abnormal model performance, check data quality")

print(f"\nFinal Test Set Results:")
print(f"  MAE:  {mae_test:.2f}")
print(f"  RMSE: {rmse_test:.2f}")
print(f"  R²:   {r2_test:.2f}")

# ========== Prediction Examples ==========
print("\n========== Test Set Prediction Examples (First 5) ==========")
prediction_df = pd.DataFrame({
    'Actual': y_test[:5],
    'Predicted': y_test_pred[:5],
    'Error': y_test[:5] - y_test_pred[:5],
    'Absolute Error': np.abs(y_test[:5] - y_test_pred[:5])
})
print(prediction_df.to_string(index=False))

# ========== Final Evaluation Metrics Summary ==========
print("\n" + "=" * 60)
print("【Final Evaluation Metrics】")
print("=" * 60)
print(f"MAE  (Mean Absolute Error):       {mae_test:.2f}")
print(f"RMSE (Root Mean Squared Error):   {rmse_test:.2f}")
print(f"R²   (R-squared Score):            {r2_test:.2f}")
print("=" * 60)

# ========== Generate Training Convergence Curve ==========
print("\n========== Generating Training Convergence Curve ==========")
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_epochs + 1), train_losses, label='Training Loss', linewidth=2, color='blue')
plt.plot(range(1, n_epochs + 1), val_losses, label='Validation Loss', linewidth=2, color='red')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('FNN Model Training Convergence Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('FNN_convergence_curve.png', dpi=300, bbox_inches='tight')
print("Convergence curve saved as: FNN_convergence_curve.png")
plt.show()

print("\n========== FNN Model Training Completed ==========")