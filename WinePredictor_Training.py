import findspark
findspark.init()

from pyspark import SparkConf
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, DecisionTree
from urllib.parse import urlparse
import boto3

def main():
    # Initialize Spark configuration and session
    conf = SparkConf().setAppName('WineQuality Training')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    # S3 Paths
    trainPath = "s3a://dataset-programming-assignment-2/TrainingDataset.csv"
    print(f"Importing: {trainPath}")

    s3ModelPath = determine_model_path(trainPath)
    print(f">>>> Model Path set: {s3ModelPath}")

    # Load and preprocess training dataset
    df_train = preprocess_training_data(spark, trainPath)

    # Train and save Decision Tree model
    print("Training DecisionTree model...")
    model_dt = DecisionTree.trainClassifier(df_train, numClasses=10, categoricalFeaturesInfo={},
                                            impurity='gini', maxDepth=10, maxBins=32)
    print("Model - DecisionTree Created")
    save_model(sc, model_dt, s3ModelPath, "model_dt.model")

    # Train and save Random Forest model
    print("Training RandomForest model...")
    model_rf = RandomForest.trainClassifier(df_train, numClasses=10, categoricalFeaturesInfo={},
                                            numTrees=10, featureSubsetStrategy="auto",
                                            impurity='gini', maxDepth=10, maxBins=32)
    print("Model - RandomForest Created")
    save_model(sc, model_rf, s3ModelPath, "model_rf.model")

    print("Data Training Completed")


def determine_model_path(trainPath):
    if not trainPath.startswith("s3://"):
        return "s3a://dataset-programming-assignment-2/models"
    return os.path.join(os.path.dirname(trainPath), "models")


def preprocess_training_data(spark, trainPath):
    # Load data from S3
    df_train = spark.read.csv(trainPath, header=True, sep=";")
    df_train.printSchema()
    df_train.show()

    # Rename columns
    df_train = df_train.withColumnRenamed('""""quality"""""', "myLabel")
    for column in df_train.columns:
        df_train = df_train.withColumnRenamed(column, column.replace('"', ''))

    # Convert data types
    for idx, col_name in enumerate(df_train.columns):
        if idx not in [6 - 1, 7 - 1, len(df_train.columns) - 1]:
            df_train = df_train.withColumn(col_name, col(col_name).cast("double"))
        elif idx in [6 - 1, 7 - 1, len(df_train.columns) - 1]:
            df_train = df_train.withColumn(col_name, col(col_name).cast("integer"))

    # Convert to RDD and LabeledPoint
    df_train = df_train.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))
    return df_train


def save_model(sc, model, s3ModelPath, model_name):
    model_path = os.path.join(s3ModelPath, model_name)
    delete_existing_model(s3ModelPath, model_name)
    model.save(sc, model_path)
    print(f">>>>> Model saved at {model_path}")


def delete_existing_model(model_path, targetFolderName):
    bucket_name = get_bucket_name(model_path)
    folder_path = f"models/{targetFolderName}"
    if folder_exists(bucket_name, folder_path):
        delete_directory(bucket_name, folder_path)


def folder_exists(bucket_name, path_to_folder):
    try:
        s3 = boto3.client('s3')
        res = s3.list_objects_v2(Bucket=bucket_name, Prefix=path_to_folder)
        return 'Contents' in res
    except Exception as e:
        print(f"An error occurred while checking folder existence: {e}")
        return False


def get_bucket_name(s3_path):
    try:
        return urlparse(s3_path).netloc
    except Exception as e:
        print(f"An error occurred while parsing bucket name: {e}")
        return None


def delete_directory(bucket_name, folder_name):
    try:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        bucket.objects.filter(Prefix=folder_name).delete()
        print(f">>>>> Pre-existing folder deleted: {folder_name}")
    except Exception as e:
        print(f"An error occurred while deleting the folder: {e}")


if __name__ == "__main__":
    main()
