# pyspark_workflow.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, upper, mean as _mean, stddev as _stddev
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Bucketizer, SQLTransformer
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import time

# Spark Session
spark = SparkSession.builder \
    .appName("EGT305_PySpark_Workflow") \
    .getOrCreate()

# load Data
df_feat = spark.read.csv("data/Employee_dataset.csv", header=True, inferSchema=True)
df_sal  = spark.read.csv("data/Employee_salaries.csv", header=True, inferSchema=True)

# normalize column names
rename_map_feat = {
    "jobId": "job_id",
    "companyId": "companyid",
    "jobRole": "jobrole",
    "education": "education",
    "major": "major",
    "industry": "industry",
    "yearsExperience": "years_experience",
    "distanceFromCBD": "distance_from_cbd"
}

rename_map_sal = {
    "jobId": "job_id",
    "salaryInThousands": "salary_in_thousands"
}

for old, new in rename_map_feat.items():
    if old in df_feat.columns:
        df_feat = df_feat.withColumnRenamed(old, new)

for old, new in rename_map_sal.items():
    if old in df_sal.columns:
        df_sal = df_sal.withColumnRenamed(old, new)

# data Cleaning

# Standardize job_id
df_feat = df_feat.withColumn("job_id", upper(col("job_id")))
df_sal  = df_sal.withColumn("job_id", upper(col("job_id")))

# remove null job_id + duplicates
df_feat = df_feat.dropna(subset=["job_id"]).dropDuplicates(["job_id"])
df_sal  = df_sal.dropna(subset=["job_id"]).dropDuplicates(["job_id"])

# join
df = df_feat.join(df_sal, on="job_id", how="inner")

# median impute for numerics
num_cols = ["years_experience", "distance_from_cbd", "salary_in_thousands"]
for c in num_cols:
    median_val = df.approxQuantile(c, [0.5], 0.001)[0]
    df = df.withColumn(c, when(col(c).isNull(), median_val).otherwise(col(c)))

# Fill categorical nulls
cat_cols = ["jobrole","education","major","industry","companyid"]
for c in cat_cols:
    df = df.withColumn(c, when(col(c).isNull(), "UNKNOWN").otherwise(col(c)))
    df = df.withColumn(c, upper(col(c)))

# Outlier capping for salary
stats = df.select(_mean("salary_in_thousands").alias("mean"),
                  _stddev("salary_in_thousands").alias("std")).collect()[0]
upper_cap = stats["mean"] + 3*stats["std"]
lower_cap = stats["mean"] - 3*stats["std"]

df = df.withColumn("salary_in_thousands",
                   when(col("salary_in_thousands") > upper_cap, upper_cap)
                   .when(col("salary_in_thousands") < lower_cap, lower_cap)
                   .otherwise(col("salary_in_thousands")))

# add CBD proximity flag now (no dependency)
df = df.withColumn("near_cbd", when(col("distance_from_cbd") < 10, 1).otherwise(0))


# feature Engineering (Pipeline)

# experience bucket
bucketizer = Bucketizer(
    splits=[-float("inf"), 5, 15, float("inf")],
    inputCol="years_experience",
    outputCol="exp_level"
)

# edu_industry interaction
edu_industry = SQLTransformer(
    statement="SELECT *, concat_ws('__', education, industry) AS edu_industry FROM __THIS__"
)

# role_exp interaction (after exp_level exists)
role_exp = SQLTransformer(
    statement="SELECT *, concat_ws('__', jobrole, CAST(exp_level AS STRING)) AS role_exp FROM __THIS__"
)

categorical_cols = ["companyid","jobrole","education","major","industry",
                    "exp_level","edu_industry","role_exp"]

indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_ohe") for c in categorical_cols]

assembler = VectorAssembler(
    inputCols=[c+"_ohe" for c in categorical_cols] + ["years_experience","distance_from_cbd","near_cbd"],
    outputCol="features_raw"
)

scaler = StandardScaler(inputCol="features_raw", outputCol="features")


# Train/Test Split
train, test = df.randomSplit([0.8, 0.2], seed=305)

# Models
models = {
    "LinearRegression": LinearRegression(featuresCol="features", labelCol="salary_in_thousands"),
    "DecisionTree": DecisionTreeRegressor(featuresCol="features", labelCol="salary_in_thousands", maxDepth=12),
    "RandomForest": RandomForestRegressor(featuresCol="features", labelCol="salary_in_thousands", numTrees=150),
    "GBT": GBTRegressor(featuresCol="features", labelCol="salary_in_thousands", maxIter=50)
}

evaluators = {
    "RMSE": RegressionEvaluator(labelCol="salary_in_thousands", predictionCol="prediction", metricName="rmse"),
    "MAE":  RegressionEvaluator(labelCol="salary_in_thousands", predictionCol="prediction", metricName="mae"),
    "R2":   RegressionEvaluator(labelCol="salary_in_thousands", predictionCol="prediction", metricName="r2")
}

# Run Experiments
results = []
for name, model in models.items():
    stages = [bucketizer, edu_industry, role_exp] + indexers + encoders + [assembler, scaler, model]
    pipeline = Pipeline(stages=stages)

    start = time.time()
    fitted = pipeline.fit(train)
    preds = fitted.transform(test)
    runtime = time.time() - start

    metrics = {m: evalr.evaluate(preds) for m, evalr in evaluators.items()}
    results.append((name, metrics["RMSE"], metrics["MAE"], metrics["R2"], runtime))

    print(f"[{name}] RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}, R²={metrics['R2']:.3f}, Time={runtime:.2f}s")

# show results
results_df = spark.createDataFrame(results, ["Model","RMSE","MAE","R2","Runtime_s"])
results_df.show(truncate=False)

# save results
results_df.coalesce(1).write.csv("results/pyspark_metrics.csv", header=True, mode="overwrite")

spark.stop()

# Comparison of PySpark and Non-PySpark Workflows

### Introduction

# This section provides a detailed comparison between the non-PySpark workflow (Pandas + scikit-learn) and the PySpark workflow (MLlib), focusing specifically on **data cleaning** and **machine learning modelling**. The comparison considers execution speed, ease of use, efficiency, and scalability, before arriving at a final recommendation tailored to this use case.

# ---

# ### Speed and Ease of Use

# In the non-PySpark workflow, data cleaning was extremely fast and straightforward. Operations such as null imputation, duplicate removal, and outlier handling were completed almost instantly due to Pandas’ in-memory processing. The syntax was concise and highly intuitive, with one-liner commands handling most cleaning tasks. Similarly, model training with scikit-learn was efficient, with Decision Tree and tuned Random Forest models completing within seconds to just over a minute. The workflow was easy to implement, debug, and extend, with a relatively gentle learning curve.

# By contrast, the PySpark workflow introduced significant overhead, even for the same dataset. Cleaning steps such as median imputation and categorical transformations required Spark jobs to execute, which increased runtime. Machine learning models trained in Spark MLlib were also slower: Random Forest required approximately 197 seconds and Gradient Boosted Trees over 316 seconds, while even Linear Regression required 57 seconds. Additionally, ease of use was reduced, since PySpark pipelines required verbose configuration with `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, and `StandardScaler`. Feature engineering steps, such as building interaction variables, involved SQL transformers, making the workflow less intuitive for beginners. Overall, while PySpark is highly powerful, its speed and ease of use are inferior to Pandas and scikit-learn for datasets of this scale.

# ---

# ### Efficiency and Scalability

# For small to moderate datasets, the non-PySpark workflow was highly efficient. Pandas and scikit-learn made full use of the single-machine CPU and RAM, delivering strong performance with minimal computational overhead. However, the workflow’s efficiency declines when datasets exceed the available memory of a single machine. This limits its scalability in large-scale industrial applications.

# PySpark, by design, was less efficient on this small dataset because of distributed job scheduling overheads. However, Spark’s architecture is built for scalability. It can distribute both data and computation across multiple executors and nodes in a cluster. Spark also provides fault tolerance and can spill data to disk when memory is insufficient, ensuring that large datasets can be processed reliably. While this overhead slows down small-scale tasks, it is critical for handling industrial-scale data volumes running into millions of rows or terabytes of information.

# ---

# ### Model Performance

# Model performance also highlighted differences between the workflows. In the non-PySpark workflow, the best-performing model was the **Random Forest (tuned)**, achieving an RMSE of 19.39, MAE of 15.66, and R² of 0.75. Other tree-based models such as Decision Tree also performed reasonably well, while regression-based models (Linear and Ridge) and boosting methods (LightGBM, XGBoost) performed poorly in this dataset.

# In the PySpark workflow, the strongest models were **Linear Regression** and **Gradient Boosted Trees (GBT)**, which both achieved R² values in the 0.73–0.74 range, with RMSE values close to 20. While these results were competitive with the non-PySpark workflow, Spark’s Random Forest underperformed significantly (R² ≈ 0.55), partly due to differences in default implementations and parameter tuning between scikit-learn and Spark MLlib. Importantly, the Spark workflow incurred significantly longer runtimes for each model compared to scikit-learn.

# ---

# ### Recommendation and Justification

# Based on this analysis, the **non-PySpark workflow (Pandas + scikit-learn) is the most suitable choice for this dataset and use case**. It provided faster execution, simpler implementation, and stronger model performance with tuned Random Forest. Its ease of use also makes it a more practical option for exploratory work and projects involving small to medium-sized datasets.

# However, it must be emphasized that **PySpark is better suited for large-scale, production-level applications** where data volumes far exceed the memory of a single machine. Spark’s distributed processing, scalability, and robustness make it the superior choice when handling millions of records or when integrated into enterprise big data platforms.

# In conclusion, the non-PySpark workflow is recommended for this specific employee salary prediction task. PySpark, while less efficient here, demonstrates scalability advantages that make it invaluable in real-world big data environments. This comparison underscores the importance of aligning workflow choice with dataset size and project requirements.


