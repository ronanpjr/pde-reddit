# src/ocr_pipeline.py
#
# Spark + EasyOCR pipeline for meme text extraction.
#
# Stages:
#   1. Load all images recursively via Spark binaryFile
#   2. Extract subreddit and filename from each path
#   3. Apply EasyOCR via a UDF (model loaded once per worker, not per row)
#   4. Join with consolidated metadata
#   5. Write output as Parquet partitioned by subreddit
#
# Usage:
#   docker compose exec jupyter python /workspace/src/ocr_pipeline.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, regexp_extract
from pyspark.sql.types import StringType
import easyocr

# ---------------------------------------------------------------------------
# Module-level singleton: initialized once per worker process, reused across
# all rows processed by that worker. Do NOT move this inside the UDF body.
# ---------------------------------------------------------------------------
_reader = None


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        # gpu=True uses whichever device CUDA_VISIBLE_DEVICES points to
        _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _reader


def extract_text(image_path: str) -> str:
    """
    Run EasyOCR on a single image and return extracted text as one string.
    Returns an empty string on any failure so the partition keeps running.
    """
    try:
        reader = get_reader()
        results = reader.readtext(image_path, detail=0)
        return " ".join(results).strip()
    except Exception:
        return ""


extract_text_udf = udf(extract_text, StringType())


def run_pipeline(images_dir: str, metadata_path: str, output_path: str) -> None:
    spark = (
        SparkSession.builder.appName("MemeOCR")
        .master("spark://spark-master:7077")
        .config("spark.executor.memory", "24g")
        .config("spark.executor.cores", "8")
        .config("spark.sql.shuffle.partitions", "32")
        .getOrCreate()
    )

    # ------------------------------------------------------------------
    # 1. Load all images recursively; filter to supported extensions only
    # ------------------------------------------------------------------
    df_images = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load(images_dir)
        .filter(
            col("path").endswith(".jpg")
            | col("path").endswith(".jpeg")
            | col("path").endswith(".png")
        )
    )

    # ------------------------------------------------------------------
    # 2. Extract subreddit name and filename from each path
    #    Expects:  /workspace/data/images/<subreddit>/<filename>
    # ------------------------------------------------------------------
    df_images = df_images.withColumn(
        "subreddit",
        regexp_extract(col("path"), r"/images/([^/]+)/", 1),
    ).withColumn(
        "filename",
        regexp_extract(col("path"), r"/([^/]+\.(?:jpe?g|png))$", 1),
    )

    # ------------------------------------------------------------------
    # 3. Run OCR — expensive step, parallelized across GPU workers
    # ------------------------------------------------------------------
    df_ocr = df_images.withColumn(
        "extracted_text",
        extract_text_udf(col("path")),
    ).select("path", "subreddit", "filename", "extracted_text")

    # ------------------------------------------------------------------
    # 4. Join with consolidated metadata on filename
    #    If the CSV uses a different column name (e.g. file_name, image_file)
    #    change the `on` parameter below accordingly.
    # ------------------------------------------------------------------
    df_meta = spark.read.csv(metadata_path, header=True, inferSchema=True)
    if "Filename" in df_meta.columns:
        df_meta = df_meta.withColumnRenamed("Filename", "filename")
        
    # Drop 'Subreddit' column from metadata to prevent ambiguous column reference error during write
    for col_name in df_meta.columns:
        if col_name.lower() == "subreddit":
            df_meta = df_meta.drop(col_name)
            
    df_final = df_ocr.join(df_meta, on="filename", how="left")

    # ------------------------------------------------------------------
    # 5. Write as Parquet partitioned by subreddit
    # ------------------------------------------------------------------
    df_final.write.mode("overwrite").partitionBy("subreddit").parquet(output_path)

    total = df_final.count()
    print(f"Pipeline complete. {total} images processed. Output: {output_path}")
    spark.stop()


if __name__ == "__main__":
    run_pipeline(
        images_dir="/workspace/data/images",
        metadata_path="/workspace/data/metadata/memes_metadata.csv",
        output_path="/workspace/data/output/ocr_results",
    )
