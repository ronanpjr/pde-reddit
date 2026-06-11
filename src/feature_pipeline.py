import argparse
import json
import os
import time
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

from feature_helpers import safe_extract_features


def extract_features_json(image_path: str, model_name: str, easyocr_model_dir: str, easyocr_download: bool) -> str:
    # wrap safe_extract_features and return JSON string
    records = safe_extract_features(image_path, model_name=model_name, easyocr_model_dir=easyocr_model_dir, easyocr_download=easyocr_download)
    return json.dumps(records)


def run_pipeline(images_dir: str, metadata_path: str, output_path: str, model_name: str = "yolov5s", easyocr_model_dir: str = "/workspace/data/easyocr_models", easyocr_download: bool = False):
    spark = (
        SparkSession.builder.appName("MemeFeatureExtraction")
        .master("spark://spark-master:7077")
        .config("spark.executor.memory", "24g")
        .config("spark.executor.cores", "8")
        .config("spark.sql.shuffle.partitions", "32")
        .getOrCreate()
    )

    # read images
    df_images = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load(images_dir)
        .filter(col("path").endswith(".jpg") | col("path").endswith(".jpeg") | col("path").endswith(".png"))
    )

    # extract subreddit and filename same as OCR pipeline
    from pyspark.sql.functions import regexp_extract

    df_images = df_images.withColumn(
        "subreddit",
        regexp_extract(col("path"), r"/images/([^/]+)/", 1),
    ).withColumn(
        "filename",
        regexp_extract(col("path"), r"/([^/]+\.(?:jpe?g|png))$", 1),
    )

    # UDF
    extract_udf = udf(lambda p: extract_features_json(p, model_name, easyocr_model_dir, easyocr_download), StringType())

    df_feat_json = df_images.select("path", "subreddit", "filename").withColumn("features_json", extract_udf(col("path")))

    # explode JSON into separate rows per detection using built-in functions: parse on driver
    # Collecting all rows to driver is not acceptable. Instead, write JSON strings to parquet and process later.
    # For simplicity, save features_json as-is in parquet; downstream job/notebook can explode and expand.

    out_path = os.path.join(output_path, "features_raw_parquet")
    df_feat_json.write.mode("overwrite").partitionBy("subreddit").parquet(out_path)

    print(f"Wrote raw features JSON to: {out_path}")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="/workspace/data/images")
    parser.add_argument("--metadata_path", default="/workspace/data/metadata_consolidated.csv")
    parser.add_argument("--output_path", default="/workspace/data/output/features")
    parser.add_argument("--model_name", default="yolov5s")
    parser.add_argument("--easyocr_model_dir", default="/workspace/data/easyocr_models")
    parser.add_argument("--easyocr_download", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    run_pipeline(args.images_dir, args.metadata_path, args.output_path, model_name=args.model_name, easyocr_model_dir=args.easyocr_model_dir, easyocr_download=args.easyocr_download)
