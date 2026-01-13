
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_data(n_samples=500):
    np.random.seed(42)
    
    # Feature: Wind Speed (Normalized 0-1 range approx for model stability)
    # Physically, this mimics "Temperature" in ZENN - the control parameter driving chaos/transition
    wind_speed = np.random.uniform(0, 1, n_samples)
    
    # Target: Damage State (Continuous proxy 0-1)
    # 0 = Undamaged, 1 = Destroyed
    # We model a sigmoidal transition where probability of damage increases with wind speed
    
    # Logistic function probability
    p_damage = 1 / (1 + np.exp(-10 * (wind_speed - 0.5)))
    
    # Sample damage states (0 or 1) then add some noise/smoothing for continuous proxy
    # In ZENN Fe3Pt example, 'PP.csv' seems to be the order parameter.
    
    # We will just use the probability curve + noise as our observable "Order Parameter"
    # This represents the "average damage state" at that wind intensity
    damage_state = p_damage + np.random.normal(0, 0.05, n_samples)
    damage_state = np.clip(damage_state, 0, 1)
    
    # Save Feature (Wind) - Format similar to V_T.csv in example (Control Parameter)
    # In Fe3Pt example: column 0 is Composition (V), column 1 is Temperature (T)
    # We will treat "Building Resistance" as V (constant or varied) and "Wind Speed" as T
    
    resistance = np.random.uniform(0.3, 0.7, n_samples) # Random structural resistance
    
    # Create DataFrame for X (Features)
    # Col 0: Resistance (V), Col 1: Wind Speed (T)
    x_df = pd.DataFrame({'Resistance': resistance, 'WindSpeed': wind_speed})
    
    # Create DataFrame for Y (Target/Order Parameter)
    y_df = pd.DataFrame({'Damage': damage_state})
    
    # Save to CSV
    x_df.to_csv('tornado_features.csv', header=False, index=False)
    y_df.to_csv('tornado_damage.csv', header=False, index=False)
    
    print(f"Generated {n_samples} samples.")
    print("Saved to tornado_features.csv and tornado_damage.csv")
    
    # Plot for sanity check
    plt.scatter(wind_speed, damage_state, alpha=0.5, s=10)
    plt.xlabel('Wind Speed (Normalized)')
    plt.ylabel('Damage State (0-1)')
    plt.title('Synthetic Tornado Damage Data')
    plt.savefig('synthetic_data_plot.png')

if __name__ == "__main__":
    generate_data()
