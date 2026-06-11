#!/usr/bin/env python3
"""Script mestre para rodar o Spark job localmente no container Jupyter/Spark."""
import subprocess
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default='/workspace/data/images')
    parser.add_argument('--metadata_path', default='/workspace/data/metadata_consolidated.csv')
    parser.add_argument('--output_path', default='/workspace/data/output/features')
    parser.add_argument('--model_name', default='yolov5s')
    parser.add_argument('--easyocr_download', action='store_true')
    args = parser.parse_args()

    cmd = [
        'python', 'src/feature_pipeline.py',
        '--images_dir', args.images_dir,
        '--metadata_path', args.metadata_path,
        '--output_path', args.output_path,
        '--model_name', args.model_name,
    ]
    if args.easyocr_download:
        cmd.append('--easyocr_download')

    print('Running:', ' '.join(cmd))
    # run and stream output
    proc = subprocess.Popen(cmd)
    proc.wait()
    if proc.returncode != 0:
        print('Job failed', proc.returncode)
        sys.exit(proc.returncode)


if __name__ == '__main__':
    main()
