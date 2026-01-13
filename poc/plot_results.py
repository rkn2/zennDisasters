
import numpy as np
import matplotlib.pyplot as plt

def plot_results():
    # Load predictions
    # Prediction (Damage State)
    y_pred = np.loadtxt("zenn_poc_predictions.txt")
    
    # Generate X axis (Wind Speed)
    # Must match the test range in zenn_tornado_poc.py
    wind_speeds = np.linspace(0.1, 1.0, 100)
    
    plt.figure(figsize=(10, 6))
    plt.plot(wind_speeds, y_pred, label="ZENN Model Prediction (Free Energy + Entropy)", color='blue', linewidth=2)
    
    # Plot expected "S-curve" if we modeled probability correctly
    # Note: Our loss was simple MSE against target data, so it should fit the data trend.
    
    plt.xlabel("Wind Speed (normalized intensity)")
    plt.ylabel("Predicted Damage State (0=None, 1=Destroyed)")
    plt.title("ZENN Proof of Concept: Tornado Damage Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = "zenn_poc_results.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_results()
