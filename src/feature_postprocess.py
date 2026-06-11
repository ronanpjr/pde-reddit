"""Post-process features_raw_parquet by exploding the JSON column into one row per detection.

Usage:
  python src/feature_postprocess.py --input /workspace/data/output/features/features_raw_parquet --output /workspace/data/output/features.parquet
"""
import argparse
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, explode, lit
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType, DoubleType, MapType


def run(input_path: str, output_path: str):
    spark = SparkSession.builder.appName('FeaturePostprocess').getOrCreate()
    df = spark.read.parquet(input_path)

    # features_json is a JSON string containing a list of records; parse it
    # We'll read it as string and use a UDF (python-side) to parse and explode
    from pyspark.sql.functions import udf
    import ast

    def parse_json(s):
        try:
            return json.loads(s) if s else []
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return []

    parse_udf = udf(lambda x: parse_json(x), ArrayType(MapType(StringType(), StringType())))

    df_parsed = df.withColumn('features_array', parse_udf(col('features_json')))
    df_exploded = df_parsed.select('path', 'subreddit', 'filename', explode(col('features_array')).alias('feat'))

    # expand map entries into columns dynamically (simple approach)
    # convert feat (map<string,string>) to JSON string and parse where needed downstream
    from pyspark.sql.functions import to_json
    df_out = df_exploded.withColumn('feat_json', to_json(col('feat')))

    df_out.write.mode('overwrite').partitionBy('subreddit').parquet(output_path)
    print('Wrote exploded features to', output_path)
    spark.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.input, args.output)
