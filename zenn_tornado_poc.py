
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# --- Configuration ---
torch.set_default_dtype(torch.double)
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device('cpu') # Forcing CPU for stability in POC on potentially unsupported MPS ops

print(f"Using device: {device}")

# --- Model Definition (Based on Fe3Pt/CZ_1D_Fe3Pt.py) ---

class Two_NN(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(2, width) # Input: [Resistance, WindSpeed]
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 1, bias=False)
        self.act = nn.Tanh() # Tanh activation
    
    def forward(self, x):
        y = self.fc1(x)
        y = self.act(y)
        y = self.fc2(y)
        y = self.act(y)
        y = self.fc3(y)
        return y

class CoupledModel(nn.Module):
    def __init__(self, width, num_networks=24, kb=0.1):
        super(CoupledModel, self).__init__()
        self.num_networks = num_networks
        self.sub_networks = nn.ModuleList([Two_NN(width) for _ in range(num_networks)])
        self.kb = kb 

    def forward(self, x):
        # x shape: [batch_size, 2] -> [Resistance, WindSpeed]
        # Treat WindSpeed as Temperature T. Clamp to avoid div by zero.
        T = x[:, 1].unsqueeze(1)
        T = torch.clamp(T, min=0.1) # Stability fix 1: Avoid very small T

        logits = []
        values = []
        sub_network_outputs = []

        # Gather outputs from all subnetworks
        for i in range(self.num_networks):
            net_out = self.sub_networks[i](x)
            sub_network_outputs.append(net_out)

        sub_network_outputs = torch.cat(sub_network_outputs, dim=1)

        # Pair networks to form Energy terms
        for i in range(self.num_networks // 2):
            net1_out = self.sub_networks[2 * i](x)      # E (Energy)
            net2_out = (self.sub_networks[2 * i + 1](x)) # S (Entropy proxy, squared for positivity)
            net2_out = net2_out**2 
            
            # Zentropy Formula components
            # value = E - T * S (Free Energy)
            value = net1_out - T * net2_out 
            
            # Logit for softmax (Probability weight) based on Boltzmann distribution
            # logit ~ -F / kT
            # Stability fix 2: Clamp value before exp
            exponent = -(value)/(self.kb*T)
            exponent = torch.clamp(exponent, max=10) # Avoid exp(large) -> inf
            
            logit = torch.exp(exponent) 
            
            logits.append(logit)
            values.append(value)

        logits = torch.cat(logits, dim=1)
        values = torch.cat(values, dim=1)

        # Weighted Sum (Expectation)
        sum_logits = torch.sum(logits, dim=1, keepdim=True) + 1e-9 # Stability fix 3
        softmax_weights = logits / sum_logits
        
        # Free Energy Expectation
        weighted_values = torch.sum(softmax_weights * values, dim=1, keepdim=True)
        
        # Entropy Expectation (from probability distribution)
        log_probs = torch.log(softmax_weights + 1e-9)
        weighted_log_probs = torch.sum(softmax_weights * log_probs, dim=1, keepdim=True)

        # Final Output: macroscopic observable
        # output = F + kT * Sum(p log p)
        output = weighted_values + self.kb * T * weighted_log_probs
        
        return output, sub_network_outputs

# --- Loss Function ---

class KLDivergenceLoss(nn.Module):
    def __init__(self, kb=0.1):
        super(KLDivergenceLoss, self).__init__()
        self.kb = kb

    def forward(self, x, y_pred, y_true):
        # Using MSE for robustness in POC
        loss = torch.mean((y_pred - y_true)**2)
        return loss

# --- Training Loop ---

def train():
    # Load Data
    X = pd.read_csv("tornado_features.csv", header=None).values
    y = pd.read_csv("tornado_damage.csv", header=None).values
    
    # Check for NaNs in input
    if np.isnan(X).any() or np.isnan(y).any():
        print("Error: NaN found in input data")
        return

    X_tensor = torch.tensor(X, dtype=torch.double).to(device)
    y_tensor = torch.tensor(y, dtype=torch.double).to(device)
    
    # Init Model
    model = CoupledModel(width=8, num_networks=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Stability fix 4: Lower LR
    criterion = KLDivergenceLoss()
    
    num_epochs = 2000
    
    print("Starting training...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred, _ = model(X_tensor)
        loss = criterion(X_tensor, y_pred, y_tensor)
        
        if torch.isnan(loss):
            print(f"Loss became NaN at epoch {epoch}")
            break
            
        loss.backward()
        
        # Stability fix 5: Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item():.6f}")

    # Generate Energy Landscape for Visualization
    # We vary WindSpeed (T) and keep Resistance (V) constant (e.g., 0.5)
    
    print("Generating landscape...")
    n_points = 100
    resistance_fixed = 0.5
    wind_speeds = np.linspace(0.1, 1.0, n_points)
    
    X_test = np.column_stack((np.full(n_points, resistance_fixed), wind_speeds))
    X_test_tensor = torch.tensor(X_test, dtype=torch.double).to(device)
    
    model.eval()
    with torch.no_grad():
        y_pred_test, sub_outputs = model(X_test_tensor)
    
    # Save outputs
    np.savetxt("zenn_poc_predictions.txt", y_pred_test.cpu().numpy())
    np.savetxt("zenn_poc_sub_outputs.txt", sub_outputs.cpu().numpy())
    print("Done. Saved results to zenn_poc_predictions.txt")

if __name__ == "__main__":
    train()
