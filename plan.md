# OCR Pipeline Implementation Plan
## Meme Dataset — Spark + EasyOCR on GPU Server

---

## Context and Goals

The dataset consists of approximately 16,000 meme images (~6GB) organized in subdirectories by subreddit, accompanied by per-subreddit metadata CSV files. Each CSV contains a filename column that maps to the image files. The goal of this first pipeline phase is to extract text from all images using EasyOCR, join the results with the metadata CSVs, and persist the output as Parquet for downstream use (CLIP embeddings, BERT analysis, unsupervised clustering).

The infrastructure is a shared GPU server running OpenSUSE with:
- 32-core Xeon CPU
- 128GB RAM
- 4x NVIDIA RTX 2080 (8GB VRAM each)
- CUDA 12.8, Driver 570.153.02
- Docker with nvidia-container-toolkit configured and verified

---

## Project Structure

```
~/meme-pipeline/
├── Dockerfile
├── docker-compose.yml
├── conf/
│   └── spark-defaults.conf
├── data/
│   ├── images/
│   │   ├── r_aww/
│   │   │   ├── abc123.jpg
│   │   │   └── def456.png
│   │   ├── r_dankmemes/
│   │   └── r_me_irl/
│   ├── metadata/
│   │   ├── r_aww.csv
│   │   ├── r_dankmemes.csv
│   │   └── r_me_irl.csv
│   └── output/
│       └── ocr_results/       # Parquet output written here
├── notebooks/
│   └── 01_ocr_validation.ipynb
└── src/
    ├── ocr_pipeline.py
    └── consolidate_metadata.py
```

---

## Step 1 — Dockerfile

Build a single image used by all services. Java is required for Spark. `libgl1` and `libglib2.0-0` are required by OpenCV, which EasyOCR depends on internally.

```dockerfile
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pyspark==3.5.1 \
    easyocr==1.7.1 \
    jupyterlab \
    pandas \
    pyarrow \
    torch \
    torchvision

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /workspace
```

---

## Step 2 — docker-compose.yml

Five services: one JupyterLab for development and validation, one Spark master, and four Spark workers with one GPU each.

All services share the same image. Each worker is assigned a dedicated GPU via `device_ids` to avoid CUDA contention. Workers are configured with 8 cores and 28GB RAM each, leaving headroom for the master and OS.

The data directory is mounted into all services so images and output are shared across the cluster without copying.

```yaml
services:
  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ~/meme-pipeline/notebooks:/workspace/notebooks
      - ~/meme-pipeline/src:/workspace/src
      - ~/meme-pipeline/data:/workspace/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    command: >
      jupyter lab --ip=0.0.0.0 --port=8888
      --no-browser --allow-root
      --NotebookApp.token='pde2026'

  spark-master:
    build: .
    ports:
      - "8080:8080"
      - "7077:7077"
    volumes:
      - ~/meme-pipeline/data:/workspace/data
    command: >
      bash -c "/opt/conda/lib/python3.10/site-packages/pyspark/sbin/start-master.sh
      && tail -f /dev/null"

  spark-worker-1:
    build: .
    depends_on: [spark-master]
    environment:
      - SPARK_WORKER_CORES=8
      - SPARK_WORKER_MEMORY=28g
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ~/meme-pipeline/data:/workspace/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    command: >
      bash -c "/opt/conda/lib/python3.10/site-packages/pyspark/sbin/start-worker.sh
      spark://spark-master:7077 && tail -f /dev/null"

  spark-worker-2:
    build: .
    depends_on: [spark-master]
    environment:
      - SPARK_WORKER_CORES=8
      - SPARK_WORKER_MEMORY=28g
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ~/meme-pipeline/data:/workspace/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    command: >
      bash -c "/opt/conda/lib/python3.10/site-packages/pyspark/sbin/start-worker.sh
      spark://spark-master:7077 && tail -f /dev/null"

  spark-worker-3:
    build: .
    depends_on: [spark-master]
    environment:
      - SPARK_WORKER_CORES=8
      - SPARK_WORKER_MEMORY=28g
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ~/meme-pipeline/data:/workspace/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['2']
              capabilities: [gpu]
    command: >
      bash -c "/opt/conda/lib/python3.10/site-packages/pyspark/sbin/start-worker.sh
      spark://spark-master:7077 && tail -f /dev/null"

  spark-worker-4:
    build: .
    depends_on: [spark-master]
    environment:
      - SPARK_WORKER_CORES=8
      - SPARK_WORKER_MEMORY=28g
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ~/meme-pipeline/data:/workspace/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['3']
              capabilities: [gpu]
    command: >
      bash -c "/opt/conda/lib/python3.10/site-packages/pyspark/sbin/start-worker.sh
      spark://spark-master:7077 && tail -f /dev/null"
```

---

## Step 3 — Metadata Consolidation

Before running the OCR pipeline, consolidate all per-subreddit CSVs into a single Spark-readable CSV. This also adds the subreddit as an explicit column, which is needed for downstream analysis.

```python
# src/consolidate_metadata.py
import pandas as pd
import os

METADATA_DIR = "/workspace/data/metadata"
OUTPUT_PATH = "/workspace/data/metadata_consolidated.csv"

dfs = []
for fname in os.listdir(METADATA_DIR):
    if not fname.endswith(".csv"):
        continue
    subreddit = fname.replace(".csv", "")
    df = pd.read_csv(os.path.join(METADATA_DIR, fname))
    df["subreddit_source"] = subreddit
    dfs.append(df)

consolidated = pd.concat(dfs, ignore_index=True)
consolidated.to_csv(OUTPUT_PATH, index=False)
print(f"Consolidated {len(dfs)} CSVs into {len(consolidated)} rows -> {OUTPUT_PATH}")
```

Run this once before submitting the Spark job:

```bash
docker compose exec jupyter python /workspace/src/consolidate_metadata.py
```

Inspect the output to confirm the filename column name. The join in the pipeline depends on it — adjust the `on` parameter in the join step if the column is named differently (e.g., `file_name`, `image_file`, `post_id`).

---

## Step 4 — OCR Pipeline

The pipeline has four stages:

1. Load all images recursively using Spark's `binaryFile` format
2. Extract subreddit and filename from each file path
3. Apply EasyOCR via a Spark UDF, running on the GPU assigned to each worker
4. Join with consolidated metadata and write output as Parquet

The EasyOCR reader is initialized once per worker using a global variable, not once per row. This is critical for performance — initializing the model inside the UDF body would reload it for every single image.

```python
# src/ocr_pipeline.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, regexp_extract
from pyspark.sql.types import StringType
import easyocr
import os

# Module-level reader: initialized once per worker process, reused across rows
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        # gpu=True uses whichever device CUDA_VISIBLE_DEVICES points to for this worker
        _reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    return _reader


def extract_text(image_path: str) -> str:
    """
    Run EasyOCR on a single image and return extracted text as a single string.
    Returns empty string on any failure to avoid killing the entire partition.
    """
    try:
        reader = get_reader()
        results = reader.readtext(image_path, detail=0)
        return " ".join(results).strip()
    except Exception:
        return ""


extract_text_udf = udf(extract_text, StringType())


def run_pipeline(images_dir: str, metadata_path: str, output_path: str):
    spark = SparkSession.builder \
        .appName("MemeOCR") \
        .master("spark://spark-master:7077") \
        .config("spark.executor.memory", "24g") \
        .config("spark.executor.cores", "8") \
        .config("spark.sql.shuffle.partitions", "32") \
        .getOrCreate()

    # Load all images recursively, filter by extension
    df_images = spark.read.format("binaryFile") \
        .option("recursiveFileLookup", "true") \
        .load(images_dir) \
        .filter(
            col("path").endswith(".jpg") |
            col("path").endswith(".jpeg") |
            col("path").endswith(".png")
        )

    # Extract subreddit name and filename from path
    # Expects paths like: /workspace/data/images/r_dankmemes/abc123.jpg
    df_images = df_images \
        .withColumn(
            "subreddit",
            regexp_extract(col("path"), r"/images/([^/]+)/", 1)
        ) \
        .withColumn(
            "filename",
            regexp_extract(col("path"), r"/([^/]+\.(?:jpe?g|png))$", 1)
        )

    # Run OCR — this is the expensive step, parallelized across workers
    df_ocr = df_images.withColumn(
        "extracted_text",
        extract_text_udf(col("path"))
    ).select("path", "subreddit", "filename", "extracted_text")

    # Load consolidated metadata and join on filename
    df_meta = spark.read.csv(metadata_path, header=True, inferSchema=True)

    # Adjust "filename" below if the metadata column has a different name
    df_final = df_ocr.join(df_meta, on="filename", how="left")

    # Write output as Parquet, partitioned by subreddit for efficient downstream reads
    df_final.write \
        .mode("overwrite") \
        .partitionBy("subreddit") \
        .parquet(output_path)

    total = df_final.count()
    print(f"Pipeline complete. {total} images processed. Output: {output_path}")
    spark.stop()


if __name__ == "__main__":
    run_pipeline(
        images_dir="/workspace/data/images",
        metadata_path="/workspace/data/metadata_consolidated.csv",
        output_path="/workspace/data/output/ocr_results"
    )
```

---

## Step 5 — Validation Notebook

Before running the full pipeline, validate the OCR on a small sample inside JupyterLab. This catches issues with image quality, encoding, or path resolution before committing to processing all 16k images.

```python
# notebooks/01_ocr_validation.ipynb

import easyocr
import matplotlib.pyplot as plt
from PIL import Image
import os

reader = easyocr.Reader(['en'], gpu=True)

# Pick 5 images from different subreddits manually
test_paths = [
    "/workspace/data/images/r_dankmemes/some_file.jpg",
    "/workspace/data/images/r_aww/another_file.png",
]

for path in test_paths:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    results = reader.readtext(path, detail=1)
    texts = [r[1] for r in results]
    confidences = [round(r[2], 2) for r in results]

    img = Image.open(path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"{os.path.basename(path)}\n{texts}", fontsize=9)
    plt.axis('off')
    plt.show()

    print(f"Extracted: {texts}")
    print(f"Confidence: {confidences}")
    print("---")
```

Also verify the path regex before running the Spark job:

```python
import re

test_path = "/workspace/data/images/r_dankmemes/abc123.jpg"

subreddit = re.search(r"/images/([^/]+)/", test_path)
filename = re.search(r"/([^/]+\.(?:jpe?g|png))$", test_path)

print("subreddit:", subreddit.group(1))  # expected: r_dankmemes
print("filename:", filename.group(1))    # expected: abc123.jpg
```

---

## Step 6 — Execution Order

```bash
# 1. Build and start all services
cd ~/meme-pipeline
docker compose up -d --build

# 2. Confirm all 4 workers registered in the Spark UI
# Open: http://localhost:8080
# Expected: 4 workers, 32 total cores, ~112GB total memory

# 3. Confirm GPUs are accessible inside a worker container
docker compose exec spark-worker-1 nvidia-smi

# 4. Open JupyterLab and run the validation notebook
# Open: http://localhost:8888  (token: pde2026)
# Run: notebooks/01_ocr_validation.ipynb

# 5. Consolidate metadata CSVs
docker compose exec jupyter python /workspace/src/consolidate_metadata.py

# 6. Inspect the consolidated CSV to confirm the filename column name
docker compose exec jupyter python -c "
import pandas as pd
df = pd.read_csv('/workspace/data/metadata_consolidated.csv', nrows=3)
print(df.columns.tolist())
print(df.head(3))
"

# 7. Submit the OCR job
docker compose exec jupyter python /workspace/src/ocr_pipeline.py

# 8. Inspect the output
docker compose exec jupyter python -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local[4]').getOrCreate()
df = spark.read.parquet('/workspace/data/output/ocr_results')
df.printSchema()
df.select('filename', 'subreddit', 'extracted_text').show(10, truncate=80)
"
```

---

## Known Risks and Mitigations

**EasyOCR accuracy on memes:** memes use non-standard fonts, text on complex backgrounds, and stylized typography. Expect a meaningful fraction of rows to have empty or garbled extracted_text. This is expected and should be documented in the analysis, not treated as a pipeline failure.

**Empty text rows:** the UDF returns an empty string on failure rather than raising an exception. After the pipeline runs, filter and count rows with `extracted_text == ""` to understand the failure rate before running downstream models.

**Metadata join misses:** if any image filename does not appear in the metadata CSVs, the left join will produce null metadata columns for that row. Check for nulls in key metadata columns after the join to detect naming mismatches between the image files and the CSV records.

**Concurrent GPU access:** the server is shared. If other users run GPU workloads simultaneously, VRAM contention will cause OOM errors inside EasyOCR. Coordinate usage with the server admin or run during off-peak hours.

**Parquet partitioning:** writing partitioned by subreddit means downstream reads can load a single subreddit efficiently. For the clustering and BERT steps, this avoids scanning the full dataset when working on one community at a time.

---

## Output Schema

After the pipeline completes, the Parquet dataset will have the following structure (metadata columns will vary depending on the original CSV schema):

| Column | Type | Description |
|---|---|---|
| filename | string | Image filename, join key |
| subreddit | string | Subreddit name extracted from path |
| path | string | Full image path on disk |
| extracted_text | string | OCR output, empty string if extraction failed |
| ... | ... | All original metadata CSV columns |

---

## Next Steps After OCR

Once the OCR Parquet is validated, the next pipeline phases are:

1. **CLIP embeddings** — load images in batch, run CLIP ViT-B/32 on each GPU worker, store embeddings as float arrays in Parquet alongside the OCR text
2. **BERT on extracted text** — filter rows where extracted_text is non-empty, tokenize, run BERT base for sentence embeddings
3. **Unsupervised clustering** — combine CLIP and BERT embeddings, run HDBSCAN, analyze cluster distribution across subreddits