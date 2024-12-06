FROM guoxiaojun2/spark-3.2.2-bin-hadoop3.2

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Install Python and required packages
USER root
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip && \
    pip3 install pyspark findspark boto3 numpy pandas scikit-learn datetime && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Ensure necessary scripts are executable
RUN chmod +x run_scripts.sh

# Switch to the default Spark user
USER ${SPARK_USER}

# Add default argument
CMD  spark-submit --master yarn cs643-programming-assignment-2/WinePredictor_Training.py
CMD  spark-submit --master yarn cs643-programming-assignment-2/WinePredictor_Eval.py
