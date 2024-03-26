# --Importing the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set seaborn styling
sns.set_theme(style="whitegrid", context="notebook", palette="pastel", font="Arial")

# --Loading the dataset
df = pd.read_csv("Customer Segmentation.csv")

# --Dropping redundant columns
df.drop(columns=["Unnamed: 0", "ID"], inplace=True)

# Data visualization
# Visualize distribution of Family Size
plt.figure(figsize=(10, 6))
sns.histplot(df["Family_Size"], bins=20, kde=True)
plt.title("Distribution of Family Size")
plt.xlabel("Family Size")
plt.ylabel("Counts")
plt.show()

# Visualize distribution of Work Experience
plt.figure(figsize=(10, 6))
sns.histplot(df["Work_Experience"], bins=20, kde=True)
plt.title("Distribution of Work Experience")
plt.xlabel("Work Experience")
plt.ylabel("Counts")
plt.show()

# Visualize distribution of Var_1
plt.figure(figsize=(10, 6))
sns.histplot(df["Var_1"])
plt.title("Distribution of Var_1")
plt.xlabel("Var_1")
plt.ylabel("Counts")
plt.show()

# Visualize distribution of Profession
plt.bar(df["Profession"].value_counts().index, df["Profession"].value_counts().values)
plt.title("Distribution of Profession")
plt.xlabel("Profession")
plt.ylabel("Counts")
plt.xticks(rotation=90)
plt.show()

# Visualize distribution of Segmentation
plt.bar(
    df["Segmentation"].value_counts().index, df["Segmentation"].value_counts().values
)
plt.title("Distribution of Segmentation")
plt.xlabel("Segmentation")
plt.ylabel("Counts")
plt.show()

# --Filling null-values using mode
df["Work_Experience"] = df["Work_Experience"].fillna(df["Work_Experience"].mode()[0])
df["Family_Size"] = df["Family_Size"].fillna(df["Family_Size"].mode()[0])

# --Dropping rows with remaining null values
df.dropna(inplace=True)

# --Label Encoding categorical values
encoder = LabelEncoder()
categorical_cols = [
    "Gender",
    "Ever_Married",
    "Graduated",
    "Profession",
    "Spending_Score",
    "Var_1",
    "Segmentation",
]
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# --Scaling the variables
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# --Plotting the Elbow Curve
SSE = []
for cluster in range(1, 15):
    model = KMeans(n_clusters=cluster, init="k-means++").fit(df_scaled)
    SSE.append(model.inertia_)

plt.figure(figsize=(15, 10))
plt.plot(range(1, 15), SSE, marker="o", c="crimson")
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title("Elbow Method for Optimal k")
plt.show()

# --Model Building
k = 4  # Selected number of clusters based on the elbow method
model = KMeans(n_clusters=k).fit(df_scaled)
labels = model.labels_

# --Getting the value counts for the different clusters
pred = model.predict(df_scaled)
df_scaled["Cluster"] = pred
plt.bar(
    df_scaled["Cluster"].value_counts().index,
    df_scaled["Cluster"].value_counts().values,
)
plt.title("Distribution of Clusters")
plt.xlabel("Cluster")
plt.ylabel("Counts")
plt.show()
