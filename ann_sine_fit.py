# Project: Neural Network with One Hidden Layer
# Function: y = sin(x)
# Run: python ann_sine_fit.py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


def main():
    # 1. Generate data for y = sin(x)
    np.random.seed(42)
    X = np.linspace(-np.pi, np.pi, 300).reshape(-1, 1)
    y = np.sin(X).ravel()

    print("Total number of samples:", len(X))

    # 2. Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 3. Scale input values using training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Create a neural network with exactly one hidden layer
    model = MLPRegressor(
        hidden_layer_sizes=(64,),
        activation="tanh",
        solver="adam",
        learning_rate_init=0.01,
        max_iter=5000,
        tol=1e-7,
        random_state=42
    )

    # 5. Train the model
    model.fit(X_train_scaled, y_train)
    print("\nModel training completed!")

    # 6. Predict test outputs and calculate MSE
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test Mean Squared Error (MSE): {mse:.8f}")

    # 7. Plot actual and predicted function values
    sorted_indices = np.argsort(X_test[:, 0])
    X_sorted = X_test[sorted_indices, 0]
    y_actual_sorted = y_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(X_sorted, y_actual_sorted, "b-", linewidth=2,
             label="Actual sin(x)")
    plt.plot(X_sorted, y_pred_sorted, "r--", linewidth=2,
             label="ANN prediction")
    plt.title("Actual vs Predicted Values of sin(x)")
    plt.xlabel("Input x (radians)")
    plt.ylabel("Output y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    plt.show()

    # 8. Plot predicted values against actual values
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, label="Test predictions")
    plt.plot([-1, 1], [-1, 1], "r--", label="Perfect prediction")
    plt.title("Predicted vs Actual Values")
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png", dpi=150)
    plt.show()

    # 9. Display a few sample predictions
    print("\nSample predictions:")
    print(f"{'Input x':>10} | {'Actual y':>10} | {'Predicted y':>12}")
    print("-" * 39)
    for i in range(min(10, len(X_test))):
        print(f"{X_test[i, 0]:10.4f} | {y_test[i]:10.4f} | {y_pred[i]:12.4f}")

    print("\nSaved graphs:")
    print("  actual_vs_predicted.png")
    print("  predicted_vs_actual.png")


if __name__ == "__main__":
    main()
