# FNN + Attention Model - Biochar Uranium Adsorption Data
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

# Set font (for Chinese characters if needed, but we'll use English)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Set random seed
torch.manual_seed(42)
np.random.seed(42)


# ========== Define Attention Layer ==========
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        # Network to learn attention weights
        self.attention_fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Calculate attention weights
        attention_weights = self.attention_fc(x)
        # Apply attention weights
        weighted_x = x * attention_weights
        return weighted_x, attention_weights


# ========== Define FNN + Attention Model ==========
class FNNWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, dropout=0.2):
        super(FNNWithAttention, self).__init__()

        # Attention layer
        self.attention = AttentionLayer(input_dim)

        # FNN layers
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # First through Attention layer
        x, attention_weights = self.attention(x)

        # Then through FNN layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x, attention_weights


# Read CSV data
data = pd.read_csv("../data/REAL_biochar_adsorption_ECs_mapped.csv")

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

# Train FNN + Attention model
print("\n========== Training FNN+Attention Model ==========")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = FNNWithAttention(input_dim=X_train_scaled.shape[1],
                         hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, dropout=0.2)
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
        outputs, _ = model(X_batch)
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
        val_outputs, _ = model(X_test_device)
        val_loss = criterion(val_outputs, y_test_device).item()
        val_losses.append(val_loss)

    # Print every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

print("Model training completed!")

# Prediction and get attention weights
model.eval()
with torch.no_grad():
    X_train_device = X_train_tensor.to(device)
    X_test_device = X_test_tensor.to(device)

    y_train_pred_tensor, train_attention = model(X_train_device)
    y_test_pred_tensor, test_attention = model(X_test_device)

    y_train_pred = y_train_pred_tensor.cpu().numpy().flatten()
    y_test_pred = y_test_pred_tensor.cpu().numpy().flatten()

    # Get average attention weights
    avg_attention_weights = test_attention.cpu().numpy().mean(axis=0)

# ========== Attention Weight Analysis ==========
print("\n========== Attention Weight Analysis ==========")
attention_df = pd.DataFrame({
    'Feature': feature_columns,
    'Attention Weight': avg_attention_weights,
    'Weight Percentage': avg_attention_weights * 100
}).sort_values('Attention Weight', ascending=False)
print(attention_df.to_string(index=False))

print(f"\nTop 3 Most Important Features:")
top3 = attention_df.head(3)
for idx, row in top3.iterrows():
    print(f"  {row['Feature']}: {row['Weight Percentage']:.2f}%")

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
print("\n========== FNN+Attention Model Information ==========")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Network structure: Attention → Input(10) → Hidden1(128) → Hidden2(64) → Hidden3(32) → Output(1)")
print(f"Dropout rate: 0.2 (20%)")
print(f"Activation function: ReLU (FNN) + Tanh (Attention)")
print(f"Attention mechanism: Learnable feature weighting")

# ========== Attention Mechanism Description ==========
print("\n========== Attention Mechanism Description ==========")
print("This model uses neural network-learned attention mechanism:")
print("1. Attention weights are learned through neural network (trainable)")
print("2. Uses Softmax to ensure weights sum to 1")
print("3. Automatically identifies important features and enhances their influence")

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

# ========== Generate Visualizations ==========
print("\n========== Generating Visualization Charts ==========")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training Convergence Curve
axes[0].plot(range(1, n_epochs + 1), train_losses, label='Training Loss', linewidth=2, color='blue')
axes[0].plot(range(1, n_epochs + 1), val_losses, label='Validation Loss', linewidth=2, color='red')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('FNN+Attention Training Convergence Curve', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Plot 2: Attention Weights
attention_sorted = attention_df.sort_values('Attention Weight', ascending=True)
axes[1].barh(attention_sorted['Feature'], attention_sorted['Attention Weight'],
             color='steelblue', edgecolor='black')
axes[1].set_xlabel('Attention Weight', fontsize=12)
axes[1].set_title('Feature Attention Weight Distribution', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../figures/FNN_Attention_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as: FNN_Attention_analysis.png")
plt.show()

print("\n========== FNN+Attention Model Training Completed ==========")