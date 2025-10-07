from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, length, regexp_extract, regexp_replace, 
    when, sum as _sum, count, lit, udf, explode, array
)
from pyspark.sql.types import StringType, IntegerType, BooleanType, ArrayType
import re

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("NameAnomalyDetection") \
    .getOrCreate()

# Sample data - Replace with your actual table
# df = spark.table("your_database.your_table")
# For demo purposes:
sample_data = [
    ("John Smith",),
    ("Mary O'Brien",),
    ("José García",),
    ("张伟",),  # Chinese characters
    ("Anne-Marie",),
    ("Dr. Johnson",),
    ("Test123",),
    ("   ",),  # Whitespace only
    ("",),  # Empty
    ("João Silva",),
    ("François Dupont",),
    ("Al-Rahman",),
    ("O'Neil-Smith",),
    ("Mr. T",),
    ("李明",),  # Chinese characters
    ("Test@User",),
    ("John..Doe",),
    ("محمد",),  # Arabic characters
    (None,),  # NULL value
]
df = spark.createDataFrame(sample_data, ["name"])

print("="*80)
print("NAME ANOMALY DETECTION ANALYSIS")
print("="*80)

# Define UDFs for character detection
def contains_chinese(text):
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def contains_arabic(text):
    if not text:
        return False
    return bool(re.search(r'[\u0600-\u06ff]', text))

def contains_cyrillic(text):
    if not text:
        return False
    return bool(re.search(r'[\u0400-\u04ff]', text))

def contains_digits(text):
    if not text:
        return False
    return bool(re.search(r'\d', text))

def contains_special_chars(text):
    if not text:
        return False
    return bool(re.search(r'[!@#$%^&*()_+=\[\]{};:"|<>?/\\]', text))

def count_periods(text):
    if not text:
        return 0
    return text.count('.')

def count_hyphens(text):
    if not text:
        return 0
    return text.count('-')

def count_apostrophes(text):
    if not text:
        return 0
    return text.count("'")

def has_title(text):
    if not text:
        return False
    titles = ['mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.']
    return any(text.lower().startswith(t) for t in titles)

# Register UDFs
udf_chinese = udf(contains_chinese, BooleanType())
udf_arabic = udf(contains_arabic, BooleanType())
udf_cyrillic = udf(contains_cyrillic, BooleanType())
udf_digits = udf(contains_digits, BooleanType())
udf_special = udf(contains_special_chars, BooleanType())
udf_periods = udf(count_periods, IntegerType())
udf_hyphens = udf(count_hyphens, IntegerType())
udf_apostrophes = udf(count_apostrophes, IntegerType())
udf_title = udf(has_title, BooleanType())

# Create analysis dataframe with all flags
analysis_df = df.withColumn("name_trimmed", 
    when(col("name").isNotNull(), regexp_replace(col("name"), r'^\s+|\s+$', ''))
    .otherwise(lit(None))
).withColumn("is_null", col("name").isNull()) \
 .withColumn("is_empty", 
    when(col("name_trimmed").isNull(), lit(True))
    .when(length(col("name_trimmed")) == 0, lit(True))
    .otherwise(lit(False))
 ) \
 .withColumn("name_length", length(col("name_trimmed"))) \
 .withColumn("has_chinese", udf_chinese(col("name_trimmed"))) \
 .withColumn("has_arabic", udf_arabic(col("name_trimmed"))) \
 .withColumn("has_cyrillic", udf_cyrillic(col("name_trimmed"))) \
 .withColumn("has_digits", udf_digits(col("name_trimmed"))) \
 .withColumn("has_special_chars", udf_special(col("name_trimmed"))) \
 .withColumn("period_count", udf_periods(col("name_trimmed"))) \
 .withColumn("hyphen_count", udf_hyphens(col("name_trimmed"))) \
 .withColumn("apostrophe_count", udf_apostrophes(col("name_trimmed"))) \
 .withColumn("has_title", udf_title(col("name_trimmed"))) \
 .withColumn("has_accents", 
    col("name_trimmed").rlike("[àáâãäåèéêëìíîïòóôõöùúûüýÿñçÀÁÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜÝŸÑÇ]")
 ) \
 .withColumn("is_purely_alphabetic",
    when(col("name_trimmed").isNull(), lit(False))
    .when(col("name_trimmed").rlike("^[a-zA-Z ]+$"), lit(True))
    .otherwise(lit(False))
 )

# Cache for performance
analysis_df.cache()

total_records = analysis_df.count()

print(f"\nTotal Records Analyzed: {total_records}")
print("="*80)

# 1. NULL and Empty Analysis
print("\n1. DATA QUALITY ISSUES")
print("-"*80)
null_empty_stats = analysis_df.agg(
    _sum(when(col("is_null"), 1).otherwise(0)).alias("null_count"),
    _sum(when(col("is_empty"), 1).otherwise(0)).alias("empty_count")
).collect()[0]

print(f"NULL values: {null_empty_stats['null_count']} ({null_empty_stats['null_count']/total_records*100:.2f}%)")
print(f"Empty/Whitespace values: {null_empty_stats['empty_count']} ({null_empty_stats['empty_count']/total_records*100:.2f}%)")

# 2. Non-Latin Script Detection
print("\n2. NON-LATIN SCRIPTS")
print("-"*80)
script_stats = analysis_df.agg(
    _sum(when(col("has_chinese"), 1).otherwise(0)).alias("chinese_count"),
    _sum(when(col("has_arabic"), 1).otherwise(0)).alias("arabic_count"),
    _sum(when(col("has_cyrillic"), 1).otherwise(0)).alias("cyrillic_count")
).collect()[0]

print(f"Chinese characters: {script_stats['chinese_count']} ({script_stats['chinese_count']/total_records*100:.2f}%)")
print(f"Arabic characters: {script_stats['arabic_count']} ({script_stats['arabic_count']/total_records*100:.2f}%)")
print(f"Cyrillic characters: {script_stats['cyrillic_count']} ({script_stats['cyrillic_count']/total_records*100:.2f}%)")

if script_stats['chinese_count'] > 0:
    print("\nSample Chinese names:")
    analysis_df.filter(col("has_chinese")).select("name").show(5, truncate=False)

if script_stats['arabic_count'] > 0:
    print("\nSample Arabic names:")
    analysis_df.filter(col("has_arabic")).select("name").show(5, truncate=False)

# 3. Special Punctuation Analysis
print("\n3. SPECIAL PUNCTUATION PATTERNS")
print("-"*80)
punct_stats = analysis_df.agg(
    _sum(when(col("period_count") > 0, 1).otherwise(0)).alias("with_periods"),
    _sum(when(col("hyphen_count") > 0, 1).otherwise(0)).alias("with_hyphens"),
    _sum(when(col("apostrophe_count") > 0, 1).otherwise(0)).alias("with_apostrophes"),
    _sum(when(col("has_title"), 1).otherwise(0)).alias("with_titles")
).collect()[0]

print(f"Names with periods (.): {punct_stats['with_periods']} ({punct_stats['with_periods']/total_records*100:.2f}%)")
print(f"Names with hyphens (-): {punct_stats['with_hyphens']} ({punct_stats['with_hyphens']/total_records*100:.2f}%)")
print(f"Names with apostrophes ('): {punct_stats['with_apostrophes']} ({punct_stats['with_apostrophes']/total_records*100:.2f}%)")
print(f"Names with titles (Mr./Dr./etc): {punct_stats['with_titles']} ({punct_stats['with_titles']/total_records*100:.2f}%)")

if punct_stats['with_periods'] > 0:
    print("\nSample names with periods:")
    analysis_df.filter(col("period_count") > 0).select("name", "period_count").show(5, truncate=False)

if punct_stats['with_hyphens'] > 0:
    print("\nSample hyphenated names:")
    analysis_df.filter(col("hyphen_count") > 0).select("name", "hyphen_count").show(5, truncate=False)

if punct_stats['with_apostrophes'] > 0:
    print("\nSample names with apostrophes:")
    analysis_df.filter(col("apostrophe_count") > 0).select("name", "apostrophe_count").show(5, truncate=False)

# 4. Unusual Character Patterns
print("\n4. UNUSUAL CHARACTER PATTERNS")
print("-"*80)
unusual_stats = analysis_df.agg(
    _sum(when(col("has_digits"), 1).otherwise(0)).alias("with_digits"),
    _sum(when(col("has_special_chars"), 1).otherwise(0)).alias("with_special"),
    _sum(when(col("has_accents"), 1).otherwise(0)).alias("with_accents"),
    _sum(when(col("is_purely_alphabetic"), 1).otherwise(0)).alias("purely_alpha")
).collect()[0]

print(f"Names with digits: {unusual_stats['with_digits']} ({unusual_stats['with_digits']/total_records*100:.2f}%)")
print(f"Names with special chars (@#$%etc): {unusual_stats['with_special']} ({unusual_stats['with_special']/total_records*100:.2f}%)")
print(f"Names with accents (é,ñ,ü,etc): {unusual_stats['with_accents']} ({unusual_stats['with_accents']/total_records*100:.2f}%)")
print(f"Purely alphabetic (a-zA-Z + space): {unusual_stats['purely_alpha']} ({unusual_stats['purely_alpha']/total_records*100:.2f}%)")

if unusual_stats['with_digits'] > 0:
    print("\nSample names with digits:")
    analysis_df.filter(col("has_digits")).select("name").show(5, truncate=False)

if unusual_stats['with_special'] > 0:
    print("\nSample names with special characters:")
    analysis_df.filter(col("has_special_chars")).select("name").show(5, truncate=False)

if unusual_stats['with_accents'] > 0:
    print("\nSample names with accents:")
    analysis_df.filter(col("has_accents")).select("name").show(5, truncate=False)

# 5. Length Analysis
print("\n5. NAME LENGTH ANALYSIS")
print("-"*80)
length_stats = analysis_df.filter(~col("is_empty")).agg(
    {"name_length": "min", "name_length": "max", "name_length": "avg"}
).collect()[0]

print(f"Shortest name: {length_stats['min(name_length)']} characters")
print(f"Longest name: {length_stats['max(name_length)']} characters")
print(f"Average length: {length_stats['avg(name_length)']:.2f} characters")

print("\nExtremely short names (1-2 characters):")
analysis_df.filter((col("name_length") <= 2) & (~col("is_empty"))).select("name", "name_length").show(5, truncate=False)

print("\nExtremely long names (>50 characters):")
analysis_df.filter(col("name_length") > 50).select("name", "name_length").show(5, truncate=False)

# 6. Combined Anomaly Summary
print("\n6. COMBINED ANOMALY DETECTION")
print("-"*80)
anomaly_df = analysis_df.withColumn("anomaly_score",
    when(col("is_null") | col("is_empty"), 10).otherwise(0) +
    when(col("has_chinese") | col("has_arabic") | col("has_cyrillic"), 3).otherwise(0) +
    when(col("has_digits"), 2).otherwise(0) +
    when(col("has_special_chars"), 4).otherwise(0) +
    when(col("period_count") > 1, 2).otherwise(0) +
    when(col("name_length") < 2, 3).otherwise(0) +
    when(col("name_length") > 50, 2).otherwise(0)
).withColumn("anomaly_category",
    when(col("anomaly_score") >= 7, "HIGH")
    .when(col("anomaly_score") >= 4, "MEDIUM")
    .when(col("anomaly_score") >= 1, "LOW")
    .otherwise("NORMAL")
)

anomaly_summary = anomaly_df.groupBy("anomaly_category").count().orderBy("anomaly_category")
print("\nAnomaly Distribution:")
anomaly_summary.show(truncate=False)

print("\nTop 10 Most Anomalous Names:")
anomaly_df.filter(col("anomaly_score") > 0) \
    .select("name", "anomaly_score", "anomaly_category") \
    .orderBy(col("anomaly_score").desc()) \
    .show(10, truncate=False)

# 7. Export flagged records (optional)
print("\n7. EXPORTING RESULTS")
print("-"*80)
print("Saving detailed analysis to: name_anomaly_analysis.parquet")
analysis_df.write.mode("overwrite").parquet("name_anomaly_analysis.parquet")

print("Saving high-priority anomalies to: high_priority_anomalies.parquet")
anomaly_df.filter(col("anomaly_category") == "HIGH") \
    .write.mode("overwrite").parquet("high_priority_anomalies.parquet")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Unpersist cached dataframe
analysis_df.unpersist()

# Stop Spark session (optional - comment out if running in notebook)
# spark.stop()
