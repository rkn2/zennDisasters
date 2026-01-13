
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# --- Configuration ---
torch.set_default_dtype(torch.double)
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device('cpu') # Force CPU for stability

print(f"Using device: {device}")

# --- Model Definition ---

class Two_NN(nn.Module):
    def __init__(self, width, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, width) 
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 1, bias=False)
        self.act = nn.Tanh() 
    
    def forward(self, x):
        y = self.fc1(x)
        y = self.act(y)
        y = self.fc2(y)
        y = self.act(y)
        y = self.fc3(y)
        return y

class CoupledModel(nn.Module):
    def __init__(self, width, input_dim, num_networks=24, kb=0.1):
        super(CoupledModel, self).__init__()
        self.num_networks = num_networks
        self.sub_networks = nn.ModuleList([Two_NN(width, input_dim) for _ in range(num_networks)])
        self.kb = kb 

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        # Last column is T (Wind Speed)
        T = x[:, -1].unsqueeze(1) 
        T = torch.clamp(T, min=0.5) # Increase min T for stability

        logits = []
        values = []
        sub_network_outputs = []

        for i in range(self.num_networks):
            net_out = self.sub_networks[i](x)
            sub_network_outputs.append(net_out)

        sub_network_outputs = torch.cat(sub_network_outputs, dim=1)

        for i in range(self.num_networks // 2):
            net1_out = self.sub_networks[2 * i](x)      # E
            net2_out = (self.sub_networks[2 * i + 1](x)) # S
            net2_out = net2_out**2 + 1e-6 # Ensure positive entropy
            
            value = net1_out - T * net2_out 
            
            exponent = -(value)/(self.kb*T)
            exponent = torch.clamp(exponent, min=-10, max=10) # Clamp both sides
            logit = torch.exp(exponent) 
            
            logits.append(logit)
            values.append(value)

        logits = torch.cat(logits, dim=1)
        values = torch.cat(values, dim=1)

        sum_logits = torch.sum(logits, dim=1, keepdim=True) + 1e-9
        softmax_weights = logits / sum_logits
        
        weighted_values = torch.sum(softmax_weights * values, dim=1, keepdim=True)
        
        log_probs = torch.log(softmax_weights + 1e-9)
        weighted_log_probs = torch.sum(softmax_weights * log_probs, dim=1, keepdim=True)

        output = weighted_values + self.kb * T * weighted_log_probs
        
        return output

class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean((y_pred - y_true)**2)
        return loss

import sys

def train(target_file="real_tornado_targets.csv", output_prefix="real"):
    # Load Real Data
    X = pd.read_csv("X_multiscale.csv", header=None).values
    y = pd.read_csv(target_file, header=None).values
    
    input_dim = X.shape[1]
    print(f"Input Dimension: {input_dim}")
    print(f"Target File: {target_file}")
    
    X_tensor = torch.tensor(X, dtype=torch.double).to(device)
    y_tensor = torch.tensor(y, dtype=torch.double).to(device)
    
    # Increase model capacity slightly more for full scale complexity
    width = 32 
    num_networks = 8 
    
    model = CoupledModel(width=width, input_dim=input_dim, num_networks=num_networks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = KLDivergenceLoss()
    
    num_epochs = 3000
    
    print("Starting training...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        
        if torch.isnan(loss):
            print(f"Loss NaN at {epoch}")
            break
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item():.6f}")

    # Save Predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)
    
    outfile = f"{output_prefix}_zenn_predictions.txt"
    np.savetxt(outfile, y_pred.cpu().numpy())
    print(f"Saved predictions to {outfile}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        output_prefix = sys.argv[2] if len(sys.argv) > 2 else "custom"
        train(target_file, output_prefix)
    else:
        train()
