import findspark
findspark.init()

from pyspark import SparkConf, SparkContext
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.tree import DecisionTreeModel, RandomForestModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import boto3


def main():
    # Initialize Spark Context and Session
    sc, spark = initialize_spark()

    # Load Pre-trained Models
    model_dt, model_rf = load_models(sc)

    # Load and Clean Validation Data
    validation_data = load_and_clean_data(spark)

    # Perform Predictions and Evaluate Models
    test_models(validation_data, model_dt, model_rf)


def initialize_spark():
    """
    Initialize SparkContext and SparkSession.
    """
    conf = SparkConf().setAppName("WineQuality Testing").set("spark.executor.cores", "1")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    return sc, spark


def load_models(sc):
    """
    Load pre-trained Decision Tree and Random Forest models from S3.
    """
    bucket_name = 'dataset-programming-assignment-2'
    model_dt_path = f"s3a://{bucket_name}/models/model_dt.model"
    model_rf_path = f"s3a://{bucket_name}/models/model_rf.model"

    # Load Decision Tree Model
    model_dt = DecisionTreeModel.load(sc, model_dt_path)
    print("Decision Tree Model Loaded")

    # Load Random Forest Model
    model_rf = RandomForestModel.load(sc, model_rf_path)
    print("Random Forest Model Loaded")

    return model_dt, model_rf


def load_and_clean_data(spark):
    """
    Load and clean the validation dataset from S3.
    """
    bucket_name = 'gayatriaavula-cs643'
    file_key = "ValidationDataset.csv"
    dataset_path = f"s3a://{bucket_name}/{file_key}"

    # Load the validation dataset
    validation = spark.read.csv(dataset_path, inferSchema=True, header=True, sep=';')
    validation = validation.withColumnRenamed('""""quality"""""', "myLabel")
    print("Quality column renamed")

    # Rename columns and cast data types
    for column in validation.columns:
        validation = validation.withColumnRenamed(column, column.replace('"', ''))

    for idx, col_name in enumerate(validation.columns):
        if idx not in [5, 6, len(validation.columns) - 1]:  # Adjust index for specific columns
            validation = validation.withColumn(col_name, col(col_name).cast("double"))
        else:
            validation = validation.withColumn(col_name, col(col_name).cast("integer"))

    print("Data cleaned, printing validation data...")
    validation.printSchema()
    validation.show()
    return validation


def test_models(validation_data, model_dt, model_rf):
    """
    Perform predictions using Decision Tree and Random Forest models,
    and evaluate their accuracy and F1 scores.
    """
    # Prepare validation data for testing
    validation_rdd = validation_data.rdd.map(lambda row: (float(row[-1]), [float(feature) for feature in row[:-1]]))
    print("Features and labels prepared for testing")

    # Decision Tree Predictions
    predictions_dt = model_dt.predict(validation_rdd.map(lambda x: x[1]))
    labels_and_predictions_dt = validation_rdd.map(lambda lp: lp[0]).zip(predictions_dt)
    print("Decision Tree predictions completed")

    # Random Forest Predictions
    predictions_rf = model_rf.predict(validation_rdd.map(lambda x: x[1]))
    labels_and_predictions_rf = validation_rdd.map(lambda lp: lp[0]).zip(predictions_rf)
    print("Random Forest predictions completed")

    # Evaluate Decision Tree
    metrics_dt = MulticlassMetrics(labels_and_predictions_dt)
    print(f"Decision Tree Model - Accuracy: {metrics_dt.accuracy:.4f}, F1 Score: {metrics_dt.weightedFMeasure():.4f}")

    # Evaluate Random Forest
    metrics_rf = MulticlassMetrics(labels_and_predictions_rf)
    print(f"Random Forest Model - Accuracy: {metrics_rf.accuracy:.4f}, F1 Score: {metrics_rf.weightedFMeasure():.4f}")

    print("Testing completed")


if __name__ == "__main__":
    main()
