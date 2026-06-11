"""Preload models to shared /workspace/data directories to avoid concurrent downloads.
Run this once inside the jupyter container before starting the Spark job.
"""
import os
from sentence_transformers import SentenceTransformer
import easyocr


def preload_easyocr(model_dir='/workspace/data/easyocr_models'):
    os.makedirs(model_dir, exist_ok=True)
    print('Preloading EasyOCR models into', model_dir)
    reader = easyocr.Reader(['en'], gpu=False, download_enabled=True, model_storage_directory=model_dir, verbose=False)
    print('EasyOCR loaded')


def preload_sentence_transformer(model_name='all-MiniLM-L6-v2'):
    print('Preloading sentence-transformers model', model_name)
    m = SentenceTransformer(model_name)
    # run a dummy encode to force download
    m.encode(['preload'], show_progress_bar=False)
    print('Sentence-transformers model ready')


def main():
    preload_easyocr()
    preload_sentence_transformer()


if __name__ == '__main__':
    main()
