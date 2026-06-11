# Pipeline distribuído de extração de features em memes

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.1-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

> **📚 Artigo base:** Sah, T., & Jordan, K. (2025). *Decoding reddit memes virality*. International Journal of Data Science and Analytics, 20:5321-5336. https://doi.org/10.1007/s41060-025-00772-5

Este repositório contém a infraestrutura e a implementação de um *job* Apache Spark focado no processamento de um *dataset* em larga escala de memes (imagens). O objetivo da arquitetura é paralelizar a extração multimodal de *features* visuais e textuais para posterior análise estatística e modelagem preditiva de viralidade, fundamentado nos conceitos do artigo supracitado.

## 🔢 Features extraídas
O pipeline aplica diversos modelos de *deep learning* e visão computacional de forma distribuída:
* **Detecção de objetos:** YOLOv5
* **Análise de cores:** Estatísticas de cor RGB/HSV extraídas por *bounding box*
* **Reconhecimento de emoções faciais:** FER (*Facial Emotion Recognition*)
* **OCR multimodal:** EasyOCR aplicado de forma global (imagem inteira) e local (recortes/crops)
* **Embeddings de texto:** Vetorização semântica via Sentence-Transformers (`all-MiniLM-L6-v2`)

## 📂 Estrutura principal do projeto

* `src/feature_helpers.py` — *Singletons* e funções de inferência encapsuladas (YOLO, EasyOCR, FER, color stats, embeddings).
* `src/feature_pipeline_stage_a.py` — Etapa A do *job* Spark: YOLO + FER + estatísticas de cor + salvamento de *crops* em disco (sem OCR/BERT).
* `src/feature_pipeline_stage_b.py` — Etapa B do *job* Spark: OCR (EasyOCR) + *embeddings* textuais (Sentence-Transformers) a partir da saída da Etapa A.
* `src/feature_pipeline.py` — Pipeline monolítico legado (mantido para compatibilidade, não recomendado para execução principal).
* `src/feature_postprocess.py` — Responsável por processar o JSON bruto, gerando uma tabela normalizada com uma linha por detecção (`features.parquet`).
* `scripts/run_two_stage_pipeline.py` — Script orquestrador recomendado: executa Etapa A e depois Etapa B com controle de recursos de GPU.
* `scripts/run_feature_pipeline.py` — Script legado para submissão simplificada do pipeline monolítico.
* `scripts/preload_models.py` — Script de execução única (*one-shot*) para pré-carregar os modelos na máquina física, evitando gargalos de *download* concorrente nos *workers*.
* `notebooks/entrega_2.ipynb` — *Notebook* de demonstração conceitual da disciplina (prova de conceito executável no Colab/Jupyter).

## ⚙️ Instalação e ambiente Docker

O `Dockerfile` inclui todas as dependências complexas de IA e big data (PyTorch, PySpark, EasyOCR, Sentence-Transformers, YOLOv5, FER, OpenCV, etc.). Recomenda-se realizar o *build* da imagem antes de subir o *cluster*.

1. **Build da imagem:**
```bash
docker compose build
```

2. **Subir os serviços (Cluster Spark + Jupyter):**
```bash
docker compose up -d
```

## ▶️ Como executar o pipeline

### Passo 1: Pré-carregamento dos modelos de IA (obrigatório)

Execute este passo uma única vez para baixar os pesos dos modelos para o diretório `/workspace/data`. Isso evita *downloads* simultâneos e concorrência nos *workers* do Spark.

```bash
docker compose exec jupyter python scripts/preload_models.py
```

*Este comando preencherá o diretório `/workspace/data/easyocr_models` e fará o cache do Sentence-Transformers.*

### Passo 2: Execução do pipeline em duas etapas (recomendado)

Recomendação operacional para evitar OOM: iniciar com menos GPUs na Etapa A (YOLO/FER) e escalar na Etapa B (OCR/BERT).

```bash
docker compose exec jupyter python scripts/run_two_stage_pipeline.py \
  --model_name yolov5n \
  --stage_a_gpus 2 \
  --stage_b_gpus 4 \
  --stage_b_repartition 16
```

*Saídas desta etapa:* 
* Etapa A (intermediário): `/workspace/data/output/features/stage_a_raw_parquet`
* *Crops* salvos: `/workspace/data/output/crops`
* Etapa B (JSON bruto para pós-processamento): `/workspace/data/output/features/features_raw_parquet`

### Passo 2.1 (opcional): execução manual por etapa

Use apenas para *debug* fino ou controle manual de cada fase.

**Etapa A — YOLO + FER + cor + crops (2 GPUs):**

```bash
docker compose exec jupyter spark-submit --master spark://spark-master:7077 \
  --conf spark.cores.max=16 \
  --conf spark.executor.cores=8 \
  --conf spark.executor.memory=24g \
  src/feature_pipeline_stage_a.py \
  --images_dir /workspace/data/images \
  --output_path /workspace/data/output/features \
  --model_name yolov5n \
  --crops_root /workspace/data/output/crops
```

**Etapa B — OCR + BERT (4 GPUs, somente após Etapa A finalizar):**

```bash
docker compose exec jupyter spark-submit --master spark://spark-master:7077 \
  --conf spark.cores.max=32 \
  --conf spark.executor.cores=8 \
  --conf spark.executor.memory=24g \
  src/feature_pipeline_stage_b.py \
  --stage_a_input /workspace/data/output/features/stage_a_raw_parquet \
  --output_path /workspace/data/output/features \
  --easyocr_model_dir /workspace/data/easyocr_models \
  --repartition 16
```

### Passo 2.2 (legado): pipeline monolítico

Disponível para compatibilidade, mas menos estável em memória GPU.

```bash
docker compose exec jupyter spark-submit --master spark://spark-master:7077 src/feature_pipeline.py \
  --images_dir /workspace/data/images \
  --metadata_path /workspace/data/metadata_consolidated.csv \
  --output_path /workspace/data/output/features \
  --model_name yolov5n
```

*Saída bruta (JSON por imagem):* `/workspace/data/output/features/features_raw_parquet`

### Passo 3: Pós-processamento (normalização de dados)

Para transformar a saída aninhada numa tabela estruturada analítica (uma linha por detecção), execute o pós-processador:

```bash
docker compose exec jupyter python src/feature_postprocess.py \
  --input /workspace/data/output/features/features_raw_parquet \
  --output /workspace/data/output/features.parquet
```

*Saída final pronta para modelagem:* `/workspace/data/output/features.parquet`

## ⚠️ Notas operacionais e boas práticas

* **Otimização do EasyOCR:** Mantenha a configuração `download_enabled=False` no ambiente de produção após realizar o Passo 1 (pré-carregamento).
* **Monitoramento:** Acompanhe a saúde do *cluster*, distribuição de tarefas e alocação de memória acessando a **Spark UI** em `http://localhost:8080`.
* **Estratégia de memória recomendada:** Execute primeiro Etapa A (YOLO/FER) e somente depois Etapa B (OCR/BERT), começando com 2 GPUs na Etapa A e escalando para 4 GPUs na Etapa B.
* **Gerenciamento de memória (OOM na GPU):** Se houver OOM, mantenha `yolov5n`, reduza `spark.cores.max`, e execute a Etapa B com menor paralelismo (`--stage_b_repartition` menor) antes de escalar novamente.
* **Escalabilidade de embeddings:** Os *embeddings* atuais estão embutidos nos JSONs como listas de *floats*. Para aplicações em produção de altíssima escala, recomenda-se externalizá-los em arquivos `.npz` e manter apenas as referências (*pointers*) no Parquet.

### Checklist pré-execução

* [ ] O *host* hospedeiro possui os drivers da NVIDIA e o `nvidia-container-toolkit` devidamente instalados?
* [ ] Os volumes no `docker-compose.yml` estão apontando para o seu diretório correto? (ex: `~/pde-reddit/data -> /workspace/data`).
* [ ] Existe espaço em disco suficiente alocado para os *downloads* dos modelos e geração das saídas descompactadas?
