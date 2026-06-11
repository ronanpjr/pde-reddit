FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-downloaded Spark tarball into image (faster, avoids network during build)
COPY assets/spark-3.5.1-bin-hadoop3.tgz /tmp/spark-3.5.1-bin-hadoop3.tgz
RUN tar -xzf /tmp/spark-3.5.1-bin-hadoop3.tgz -C /opt && \
    rm /tmp/spark-3.5.1-bin-hadoop3.tgz

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
