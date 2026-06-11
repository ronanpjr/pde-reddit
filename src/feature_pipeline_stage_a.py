import argparse
import json
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, udf
from pyspark.sql.types import StringType

from feature_helpers import safe_extract_stage_a


def extract_stage_a_json(image_path: str, model_name: str, crops_root: str) -> str:
    records = safe_extract_stage_a(
        image_path=image_path,
        model_name=model_name,
        crops_root=crops_root,
        prefer_gpu=True,
        oom_fallback_to_cpu=True,
    )
    return json.dumps(records)


def run_pipeline(images_dir: str, output_path: str, model_name: str = "yolov5n", crops_root: str = "/workspace/data/output/crops"):
    spark = (
        SparkSession.builder.appName("MemeFeatureStageA")
        .master("spark://spark-master:7077")
        .config("spark.executor.memory", "24g")
        .config("spark.executor.cores", "8")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )

    df_images = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load(images_dir)
        .filter(col("path").endswith(".jpg") | col("path").endswith(".jpeg") | col("path").endswith(".png"))
    )

    df_images = df_images.withColumn("subreddit", regexp_extract(col("path"), r"/images/([^/]+)/", 1)).withColumn(
        "filename", regexp_extract(col("path"), r"/([^/]+\.(?:jpe?g|png))$", 1)
    )

    extract_udf = udf(lambda p: extract_stage_a_json(p, model_name, crops_root), StringType())
    df_stage_a = df_images.select("path", "subreddit", "filename").withColumn("stage_a_json", extract_udf(col("path")))

    out_path = os.path.join(output_path, "stage_a_raw_parquet")
    df_stage_a.write.mode("overwrite").partitionBy("subreddit").parquet(out_path)
    print(f"Wrote Stage A raw JSON to: {out_path}")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="/workspace/data/images")
    parser.add_argument("--output_path", default="/workspace/data/output/features")
    parser.add_argument("--model_name", default="yolov5n")
    parser.add_argument("--crops_root", default="/workspace/data/output/crops")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.crops_root, exist_ok=True)
    run_pipeline(args.images_dir, args.output_path, model_name=args.model_name, crops_root=args.crops_root)
