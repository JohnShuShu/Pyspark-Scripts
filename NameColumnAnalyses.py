from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, size, when, length, mean, stddev, sum, lit, count
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Initialize Spark Session (for local testing)
# In a cluster environment (e.g., Databricks), the SparkSession is usually pre-initialized.
try:
    spark = SparkSession.builder.appName("NameAnalytics").getOrCreate()
    print("Spark Session Initialized.")
except Exception as e:
    print(f"Could not initialize Spark Session: {e}")
    # Fallback/exit if Spark fails
    # exit(1)

# --- 1. Data Setup (Simulating a Table Load) ---
# Replace this with your actual table loading logic (e.g., spark.read.table("your_database.your_table"))

data = [
    ("Alexander", "The", "Great", 30),
    ("Marie", "Sklodowska", "Curie", 45),
    ("Albert", None, "Einstein", 76),
    ("Ada", "Lovelace", "Byron", 36),
    ("Galileo", None, "Galilei", 77),
    ("Isaac", "New ton", "Newton", 84), # Intentional space in middle name
    ("Nikola", None, "Tes la", 86),      # Intentional space in last name
    ("Charles", "R", "Darwin", 73),
    ("Alan", "Mathison", "Turing", 41),
    ("Alan", "M.", "Turing", 41),        # Short Middle Initial
    ("Firstname", "Loooooooooooooong", "Lastname", 50), # Outlier length
    ("Tiny", None, "Xu", 22),            # Short name
    ("John", "F.", "Kennedy", 46),
    ("Mary", "Jane", "Watson", 28)
]
columns = ["FirstName", "MiddleName", "LastName", "Age"]
df = spark.createDataFrame(data, columns)

df.printSchema()
df.show(5, truncate=False)

# --- 2. Data Cleaning and Feature Engineering ---

print("\n--- 2. Feature Engineering and Name Cleaning ---")

name_analysis_df = df.withColumn(
    "Name_has_space",
    # Check for spaces in First, Middle, or Last names (Outlier/Bad Data pattern)
    (col("FirstName").contains(" ")) | 
    (col("MiddleName").contains(" ")) | 
    (col("LastName").contains(" "))
).withColumn(
    "FName_Length", length(col("FirstName"))
).withColumn(
    "MName_Length", length(col("MiddleName"))
).withColumn(
    "LName_Length", length(col("LastName"))
).withColumn(
    "Is_Initial_MName", 
    # Identify Middle Initials (potential pattern/deviation)
    (length(col("MiddleName")) == 2) & 
    (col("MiddleName").like("_.%")) 
)

name_analysis_df.select("FirstName", "MiddleName", "LastName", "FName_Length", "Name_has_space", "Is_Initial_MName").show(10)

# --- 3. Data Science Analytics (Statistical & Outlier Detection) ---

print("\n--- 3. Statistical Analytics & Outlier Detection ---")

# a) Basic Descriptive Statistics on Name Lengths
# Aggregate statistics for finding deviation and central tendency
name_stats = name_analysis_df.select(
    mean("FName_Length").alias("Avg_FName_Len"),
    stddev("FName_Length").alias("StdDev_FName_Len"),
    mean("LName_Length").alias("Avg_LName_Len"),
    stddev("LName_Length").alias("StdDev_LName_Len")
).collect()[0].asDict()

print(f"Average First Name Length: {name_stats['Avg_FName_Len']:.2f} (StdDev: {name_stats['StdDev_FName_Len']:.2f})")
print(f"Average Last Name Length: {name_stats['Avg_LName_Len']:.2f} (StdDev: {name_stats['StdDev_LName_Len']:.2f})")

# b) Identifying Length Outliers (Intuitive Outliers: names significantly longer/shorter than average)
# Let's define an outlier as a name length > Avg + 2*StdDev (a common heuristic)

# Calculate Upper Bound for First Name Outlier
f_name_outlier_bound = name_stats['Avg_FName_Len'] + 2 * name_stats['StdDev_FName_Len']
l_name_outlier_bound = name_stats['Avg_LName_Len'] + 2 * name_stats['StdDev_LName_Len']

outliers_df = name_analysis_df.filter(
    (col("FName_Length") > f_name_outlier_bound) | 
    (col("LName_Length") > l_name_outlier_bound)
).select("FirstName", "LastName", "FName_Length", "LName_Length").distinct()

print(f"\nPotential Name Length Outliers (First Name > {f_name_outlier_bound:.2f} or Last Name > {l_name_outlier_bound:.2f}):")
outliers_df.show(truncate=False)

# c) Phenomenon: Names with Spaces (Data Quality/Pattern Deviation)
space_names_df = name_analysis_df.filter(col("Name_has_space")).select("FirstName", "MiddleName", "LastName")
print("\n--- Phenomenon: Names with Spaces (Potential Data Entry Errors) ---")
space_names_df.show(truncate=False)

# d) Phenomenon: Middle Initial vs. Full Middle Name
initial_count = name_analysis_df.filter(col("Is_Initial_MName")).count()
full_name_count = name_analysis_df.filter(
    (col("MiddleName").isNotNull()) & (~col("Is_Initial_MName"))
).count()
null_mname_count = name_analysis_df.filter(col("MiddleName").isNull()).count()

print(f"\n--- Phenomenon: Middle Name Patterns ---")
print(f"Count of Middle Initials: {initial_count}")
print(f"Count of Full Middle Names: {full_name_count}")
print(f"Count of Missing Middle Names: {null_mname_count}")


# --- 4. Prepare Data for Visualization (Collect to Pandas) ---

# Histogram Data for First Name Lengths
f_name_lengths_pd = name_analysis_df.select("FName_Length").toPandas()

# Bar Chart Data for Middle Name Patterns
mname_pattern_data = pd.DataFrame({
    'Pattern': ['Middle Initial', 'Full Middle Name', 'Missing Middle Name'],
    'Count': [initial_count, full_name_count, null_mname_count]
})


# --- 5. Data Visualization (using Pandas/Matplotlib) ---

print("\n--- 5. Data Visualization ---")

plt.style.use('seaborn-v0_8-whitegrid')

# a) Histogram for First Name Lengths (Distribution Insight)
plt.figure(figsize=(10, 6))
sns.histplot(f_name_lengths_pd['FName_Length'], kde=True, bins=range(f_name_lengths_pd['FName_Length'].min(), f_name_lengths_pd['FName_Length'].max() + 2), color='skyblue')
plt.title('Distribution of First Name Lengths')
plt.xlabel('First Name Length (Characters)')
plt.ylabel('Frequency')
plt.xticks(range(f_name_lengths_pd['FName_Length'].min(), f_name_lengths_pd['FName_Length'].max() + 1))
plt.show() # 

# b) Bar Plot for Middle Name Patterns (Intuitive Insight)
plt.figure(figsize=(8, 5))
sns.barplot(x='Pattern', y='Count', data=mname_pattern_data, palette='viridis')
plt.title('Middle Name Usage Pattern')
plt.xlabel('Middle Name Type')
plt.ylabel('Count')
plt.show() # 

# Stop Spark Session
spark.stop()
