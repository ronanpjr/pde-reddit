==============================================================
Este repositório contém a implementação de um Spark Job que processa um dataset de memes (imagens) e extrai features para análise estatística:

- Detecção de objetos com YOLOv5
- Estatísticas de cor por bounding box
- Reconhecimento de emoções faciais (FER)
- OCR com EasyOCR (crop + global)
- Embeddings de texto via Sentence-Transformers (all-MiniLM-L6-v2)

Arquivos principais adicionados
-----------------------------
- src/feature_helpers.py — singletons e funções de inferência (YOLO, EasyOCR, FER, color stats, embeddings).
- src/feature_pipeline.py — Spark job que executa a extração por imagem e grava features_raw_parquet (JSON por imagem).
- src/feature_postprocess.py — pós-processador que explode features_json para uma tabela por detecção (features.parquet).
- scripts/run_feature_pipeline.py — script mestre para executar o job.
- scripts/preload_models.py — script one-shot para pré-carregar modelos e evitar downloads concorrentes.
- notebooks/entrega_2.ipynb — notebook inicial (pipeline em Colab) (mantido no repositório).

Dependências e imagem Docker
----------------------------
O Dockerfile já foi atualizado para incluir as dependências necessárias (PyTorch, pyspark, easyocr, sentence-transformers, yolov5, fer, opencv, etc.). Recomenda-se rebuildar a imagem antes de subir os serviços.

1) Build da imagem

  docker compose build

2) Subir a stack

  docker compose up -d

Pré-carregamento de modelos (recomendado)
--------------------------------------
Execute uma vez para baixar modelos em /workspace/data e evitar downloads simultâneos nos workers:

  docker compose exec jupyter python scripts/preload_models.py

Isso vai preencher /workspace/data/easyocr_models e baixar o modelo sentence-transformers para o cache.

Execução do job Spark
---------------------
Opção A — script mestre (simples):

  docker compose exec jupyter python scripts/run_feature_pipeline.py

Opção B — spark-submit (recomendado para cluster):

  docker compose exec jupyter spark-submit --master spark://spark-master:7077 src/feature_pipeline.py \
    --images_dir /workspace/data/images \
    --metadata_path /workspace/data/metadata_consolidated.csv \
    --output_path /workspace/data/output/features

Saída inicial
-------------
O job gera um Parquet com uma linha por imagem contendo uma coluna features_json (JSON string com lista de records por detecção):

  /workspace/data/output/features/features_raw_parquet

Pós-processamento (explodir JSON em linhas por detecção)
----------------------------------------------------
Para transformar a saída em uma tabela com uma linha por detecção e schema normalizado rode:

  docker compose exec jupyter python src/feature_postprocess.py --input /workspace/data/output/features/features_raw_parquet --output /workspace/data/output/features.parquet

Agora a saída estará em:

  /workspace/data/output/features.parquet

Notas operacionais e boas práticas
--------------------------------
- Pré-carregue modelos para evitar downloads concorrentes (scripts/preload_models.py).\n+- Mantenha download_enabled=False no EasyOCR em produção, após pré-carregamento.\n+- Monitore o Spark UI em http://localhost:8080 para verificar executores e tarefas.\n+- Se houver OOM na GPU, reduza o tamanho máximo dos crops ou rode embeddings na CPU.\n+- Os embeddings estão incluídos nos JSON como listas de floats por padrão. Para produção em larga escala é recomendável externalizar os embeddings em arquivos .npz e manter referência no Parquet.

Verificações antes de rodar
--------------------------
1. Confirme que o host tem drivers NVIDIA e nvidia-container-toolkit instalados.\n2. Verifique se os volumes em docker-compose.yml apontam para os diretórios corretos (~/pde-reddit/data -> /workspace/data).\n3. Confirme espaço em disco suficiente para modelos e outputs.

