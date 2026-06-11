FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and install full Apache Spark 3.5.1 (includes sbin cluster scripts)
RUN wget -q https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz && \
    tar -xzf spark-3.5.1-bin-hadoop3.tgz -C /opt && \
    rm spark-3.5.1-bin-hadoop3.tgz

ENV SPARK_HOME=/opt/spark-3.5.1-bin-hadoop3
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${JAVA_HOME}/bin:${PATH}"

RUN pip install --no-cache-dir \
    pyspark==3.5.1 \
    easyocr==1.7.1 \
    jupyterlab \
    pandas \
    pyarrow \
    torch \
    torchvision \
    sentence-transformers \
    yolov5 \
    fer \
    opencv-python-headless \
    pillow \
    tqdm \
    matplotlib \
    mtcnn

WORKDIR /workspace
