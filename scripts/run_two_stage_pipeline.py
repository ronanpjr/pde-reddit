#!/usr/bin/env python3
import argparse
import subprocess
import sys


def run(cmd):
    print("Running:", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def spark_submit(module_path: str, args_list: list[str], max_cores: int):
    cmd = [
        "spark-submit",
        "--master",
        "spark://spark-master:7077",
        "--conf",
        f"spark.cores.max={max_cores}",
        "--conf",
        "spark.executor.cores=8",
        "--conf",
        "spark.executor.memory=24g",
        module_path,
    ]
    cmd.extend(args_list)
    run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run two-stage feature pipeline (YOLO/FER then OCR/BERT)")
    parser.add_argument("--images_dir", default="/workspace/data/images")
    parser.add_argument("--output_path", default="/workspace/data/output/features")
    parser.add_argument("--model_name", default="yolov5n")
    parser.add_argument("--crops_root", default="/workspace/data/output/crops")
    parser.add_argument("--easyocr_model_dir", default="/workspace/data/easyocr_models")
    parser.add_argument("--easyocr_download", action="store_true")
    parser.add_argument("--stage_a_gpus", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--stage_b_gpus", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--stage_b_repartition", type=int, default=16)
    parser.add_argument("--skip_stage_a", action="store_true")
    parser.add_argument("--skip_stage_b", action="store_true")
    args = parser.parse_args()

    stage_a_input = f"{args.output_path.rstrip('/')}/stage_a_raw_parquet"

    try:
        if not args.skip_stage_a:
            stage_a_cores = args.stage_a_gpus * 8
            spark_submit(
                "src/feature_pipeline_stage_a.py",
                [
                    "--images_dir",
                    args.images_dir,
                    "--output_path",
                    args.output_path,
                    "--model_name",
                    args.model_name,
                    "--crops_root",
                    args.crops_root,
                ],
                max_cores=stage_a_cores,
            )

        if not args.skip_stage_b:
            stage_b_cores = args.stage_b_gpus * 8
            stage_b_args = [
                "--stage_a_input",
                stage_a_input,
                "--output_path",
                args.output_path,
                "--easyocr_model_dir",
                args.easyocr_model_dir,
                "--repartition",
                str(args.stage_b_repartition),
            ]
            if args.easyocr_download:
                stage_b_args.append("--easyocr_download")

            spark_submit("src/feature_pipeline_stage_b.py", stage_b_args, max_cores=stage_b_cores)

    except Exception as exc:
        print("Pipeline failed:", exc)
        sys.exit(1)

    print("Two-stage pipeline completed successfully")


if __name__ == "__main__":
    main()
