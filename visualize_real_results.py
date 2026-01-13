
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualize():
    # Load Data
    X = pd.read_csv("real_tornado_features.csv", header=None)
    y_true = pd.read_csv("real_tornado_targets.csv", header=None)
    y_pred = np.loadtxt("real_zenn_predictions.txt")
    
    # Extract T (Last column of X)
    # Recall T was (EF + 1) / 6.0
    T = X.iloc[:, -1].values
    
    # Extract EF for plotting axis (Reverse normalization)
    EF = T * 6.0 - 1.0
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of True Data
    plt.scatter(EF, y_true, color='gray', alpha=0.5, label='True Damage (Observed)')
    
    # Scatter plot of Predictions
    plt.scatter(EF, y_pred, color='blue', alpha=0.7, label='ZENN Prediction', marker='x')
    
    plt.xlabel("Tornado Intensity (EF Scale)")
    plt.ylabel("Damage State (Normalized 0-1)")
    plt.title("ZENN Model: Real Tornado Data Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = "real_zenn_results.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    visualize()
