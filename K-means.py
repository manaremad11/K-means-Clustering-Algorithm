# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Step 3: Explore the data
print(data.head())  # Display the first few rows
print(data.info())  # Check the structure and for missing values

# Select relevant columns for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Data Preprocessing (Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Determine the optimal number of clusters using the Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Step 6: Applying K-means with optimal clusters (say, 5 based on Elbow Method)
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 7: Visualize the clusters
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=data['Cluster'], cmap='viridis')
plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.colorbar()
plt.show()
