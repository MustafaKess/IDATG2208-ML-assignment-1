# Exercise 1.2 Correlation Analysis
# Q1.2.1 Compute the correlation matrix for all features in the dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("archive/WineQT.csv")
correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)

#Q1.2.2 Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Wine Features")
plt.show()