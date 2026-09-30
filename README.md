# Ai-based-load-forecasting
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample historical load data (in MW)
data = {
    "Hour": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Load": [120, 125, 130, 128, 135, 145, 155, 165, 170, 180]
}

df = pd.DataFrame(data)

# Input and output
X = df[["Hour"]]
y = df["Load"]

# Create AI model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Forecast future load
future_hours = np.array([[11], [12], [13], [14], [15]])

predicted_load = model.predict(future_hours)

# Display results
print("AI-Based Load Forecasting")
print("-------------------------")

for hour, load in zip(future_hours.flatten(), predicted_load):
    print(f"Hour {hour}: Predicted Load = {load:.2f} MW")

# Plot actual and predicted load
plt.scatter(df["Hour"], df["Load"], label="Actual Load")
plt.plot(df["Hour"], model.predict(X), label="Predicted Trend")

plt.xlabel("Hour")
plt.ylabel("Load (MW)")
plt.title("AI-Based Electrical Load Forecasting")
plt.legend()
plt.grid()
plt.show()
