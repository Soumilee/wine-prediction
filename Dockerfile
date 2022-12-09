From datamechanics/spark:2.4.7-hadoop-3.1.0-java-8-scala-2.12-python-3.7-dm18

ENV PYSPARK_MAJOR_PYTHON_VERSION=3
RUN conda install -y numpy
RUN conda install -y pandas
RUN conda install -y py4j

WORKDIR /opt/wine-prediction-app

COPY model_prediction.py .
ADD datamodel/ValidationDataset.csv .
ADD datamodel ./datamodel/
