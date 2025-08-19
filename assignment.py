
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Task 1: Load and Explore the Dataset
# -----------------------------
try:
    # You can replace this with another dataset if you want
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame  # Pandas DataFrame

    print("Dataset loaded successfully!\n")

    # Display first few rows
    print("First 5 rows of dataset:")
    print(df.head(), "\n")

    # Explore structure
    print("Dataset Info:")
    print(df.info(), "\n")

    # Check for missing values
    print("Missing values per column:")
    print(df.isnull().sum(), "\n")

    # Clean dataset (here: iris has no missing values, but for demo we show how)
    df = df.dropna()  # Alternatively, df.fillna(method='ffill')

except FileNotFoundError:
    print("Error: Dataset file not found. Please check the file path.")
except Exception as e:
    print("Error loading dataset:", str(e))


# -----------------------------
# Task 2: Basic Data Analysis
# -----------------------------
print("Basic Statistics of Numerical Columns:")
print(df.describe(), "\n")

# Grouping example: Mean of features per species
grouped = df.groupby("target").mean()
print("Mean values grouped by Species (target):")
print(grouped, "\n")

# Observations
print("Observations:")
print("- Sepal length tends to increase with species index.")
print("- Petal measurements differ strongly by species, useful for classification.\n")


# -----------------------------
# Task 3: Data Visualization
# -----------------------------

# 1. Line chart (trends over sample index)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.plot(df.index, df["petal length (cm)"], label="Petal Length")
plt.title("Line Chart: Sepal vs Petal Length Trend")
plt.xlabel("Sample Index")
plt.ylabel("Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart (average petal length per species)
plt.figure(figsize=(6,4))
sns.barplot(x="target", y="petal length (cm)", data=df, ci=None)
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species (Target)")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(6,4))
plt.hist(df["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram: Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (relationship between sepal length and petal length)
plt.figure(figsize=(6,4))
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"], c=df["target"], cmap="viridis")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.colorbar(label="Species")
plt.show()

print("Visualization complete!")
