from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, isnan, to_timestamp, concat_ws
from pyspark.sql.functions import avg
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import NumericType

import sys

spark = SparkSession.builder.appName("AirQualityPreprocessing").getOrCreate()

csv_path= "/home/sat3812/Downloads/Project1/air+quality/AirQualityUCI.csv"
out_parquet = "/home/sat3812/Downloads/Project1/air+quality/AirQuality_clean.parquet"
out_dir = "/home/sat3812/Downloads/Project1/air+quality/analysis_results"

df = spark.read.option("header", True).option("sep", ";").option("inferSchema", True).csv(csv_path)

# Replace spaces, dots, and parentheses in column names with underscores
for c in df.columns:
    clean_c = c.replace("(", "_").replace(")", "_").replace(".", "_").replace(" ", "_")
    df = df.withColumnRenamed(c, clean_c)

df.printSchema()

cols = df.columns

numeric_cols = [f.name for f in df.schema.fields if str(f.dataType) in ("IntegerType", "DoubleType", "LongType", "FloatType")]
# Some numeric fields might be strings (if commas exist), convert them where necessary
for c in numeric_cols:
    df = df.withColumn(c, when(col(c) == -200, None).otherwise(col(c)))
    
def try_cast(df, colname):
    try:
        return df.withColumn(colname, col(colname).cast("double"))
    except:
        return df

for c in numeric_cols:
    df = try_cast(df, c)
    
# Basic dropping of columns with majority missing (>60%) - adjust threshold
threshold = 0.6
total = df.count()
drop_cols = []
for c in numeric_cols:
    nulls = df.filter(col(c).isNull()).count()
    if nulls / total > threshold:
        drop_cols.append(c)

if drop_cols:
    print("Dropping columns with many missing values:", drop_cols)
    df = df.drop(*drop_cols)

# --- Replace -200 values with null and cast to double ---
numeric_cols = ["CO_GT_", "PT08_S1_CO_", "NMHC_GT_", "C6H6_GT_",
                "NOx_GT_", "NO2_GT_", "T", "RH", "AH"]

for c in numeric_cols:
    df = df.withColumn(c, when(col(c) == -200, None).otherwise(col(c).cast("double")))

non_null_cols = [c for c in df.columns if df.filter(col(c).isNotNull()).count() > 0]
df = df.select(non_null_cols)

print("Columns retained after dropping fully-null ones:", non_null_cols)

# Now Imputing the numeric columns using median

imputer = Imputer(strategy="median",
                  inputCols=[c for c in numeric_cols if c in df.columns],
                  outputCols=[c for c in numeric_cols if c in df.columns])
df = imputer.fit(df).transform(df)
df.show()

# Save cleaned data as Parquet for fast read by analysis step
df.write.mode("overwrite").parquet(out_parquet)
spark.catalog.clearCache()

print("Preprocessing finished. Saved to:", out_parquet)

# Give Spark a short moment to finalize file commits
import time
time.sleep(2)

print("\n Performing Correlation Analysis...")

pollutant_cols = [
    "CO_GT_",
    "PT08_S1_CO_",
    "NMHC_GT_",
    "C6H6_GT_",
    "NOx_GT_",
    "NO2_GT_",
    "T",
    "RH",
    "AH",
]

# Safely reload cleaned data (read entire directory, not a single file)
df_clean = spark.read.parquet(out_parquet)

# Verify data presence
print(f"Loaded {df_clean.count()} rows from cleaned parquet file.")

# Drop rows with nulls in pollutant columns
df_clean = df_clean.na.drop(subset=[c for c in pollutant_cols if c in df_clean.columns])
print(f"Rows remaining after dropping nulls in pollutants: {df_clean.count()}")

# Assemble features for correlation
assembler_corr = VectorAssembler(
    inputCols=[c for c in pollutant_cols if c in df_clean.columns],
    outputCol="features_vec",
    handleInvalid="skip",
)
vec_df = assembler_corr.transform(df_clean)

# Compute correlation matrix
corr_matrix = Correlation.corr(vec_df, "features_vec", "pearson").head()[0]
print("Correlation matrix computed successfully!\n")
print(corr_matrix)

# ---- LINEAR REGRESSION (Predicting CO(GT)) ----
if "CO_GT_" in df_clean.columns:
    features = [c for c in pollutant_cols if c in df_clean.columns and c != "CO_GT_"]
    assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip")
    lr_df = assembler.transform(df_clean).select("features", col("CO_GT_").alias("label"))

    train, test = lr_df.randomSplit([0.8, 0.2], seed=42)

    lr = LinearRegression(featuresCol="features", labelCol="label")
    lr_model = lr.fit(train)
    preds = lr_model.transform(test)

    evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    print(f"Linear Regression RMSE: {evaluator_rmse.evaluate(preds)}")
    print(f"Linear Regression R2: {evaluator_r2.evaluate(preds)}")
    print("Coefficients:", lr_model.coefficients)
    print("Intercept:", lr_model.intercept)

spark.stop()
