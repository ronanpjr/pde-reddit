import argparse
import json
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

from feature_helpers import enrich_stage_a_records_with_text


def enrich_json(path: str, filename: str, stage_a_json: str, easyocr_model_dir: str, easyocr_download: bool) -> str:
    try:
        stage_a_records = json.loads(stage_a_json) if stage_a_json else []
    except Exception:
        stage_a_records = []
    out = enrich_stage_a_records_with_text(
        image_path=path,
        filename=filename,
        stage_a_records=stage_a_records,
        easyocr_model_dir=easyocr_model_dir,
        easyocr_download=easyocr_download,
    )
    return json.dumps(out)


def run_pipeline(
    stage_a_input: str,
    output_path: str,
    easyocr_model_dir: str = "/workspace/data/easyocr_models",
    easyocr_download: bool = False,
    repartition: int = 16,
):
    spark = (
        SparkSession.builder.appName("MemeFeatureStageB")
        .master("spark://spark-master:7077")
        .config("spark.executor.memory", "24g")
        .config("spark.executor.cores", "8")
        .config("spark.sql.shuffle.partitions", str(repartition))
        .getOrCreate()
    )

    df = spark.read.parquet(stage_a_input)
    if repartition and repartition > 0:
        df = df.repartition(repartition)

    enrich_udf = udf(lambda p, f, s: enrich_json(p, f, s, easyocr_model_dir, easyocr_download), StringType())
    df_stage_b = df.select("path", "subreddit", "filename", "stage_a_json").withColumn(
        "features_json", enrich_udf(col("path"), col("filename"), col("stage_a_json"))
    )

    out_path = os.path.join(output_path, "features_raw_parquet")
    df_stage_b.select("path", "subreddit", "filename", "features_json").write.mode("overwrite").partitionBy("subreddit").parquet(out_path)

    print(f"Wrote Stage B features JSON to: {out_path}")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_a_input", default="/workspace/data/output/features/stage_a_raw_parquet")
    parser.add_argument("--output_path", default="/workspace/data/output/features")
    parser.add_argument("--easyocr_model_dir", default="/workspace/data/easyocr_models")
    parser.add_argument("--easyocr_download", action="store_true")
    parser.add_argument("--repartition", type=int, default=16)
    args = parser.parse_args()

    run_pipeline(
        stage_a_input=args.stage_a_input,
        output_path=args.output_path,
        easyocr_model_dir=args.easyocr_model_dir,
        easyocr_download=args.easyocr_download,
        repartition=args.repartition,
    )
