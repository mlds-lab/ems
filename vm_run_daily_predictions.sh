#!/bin/bash

export PYSPARK_PYTHON=/usr/bin/python3.6
export SPARK_HOME=/usr/local/spark/
export PATH=$SPARK_HOME/bin:$PATH
export SPARK_MASTER="local[*]"

export PYSPARK_DRIVER_PYTHON=/usr/bin/python3.6
export PYTHONPATH=/usr/local/spark/python/lib/py4j-0.10.4-src.zip:/usr/local/spark/python/

sudo chmod 777 /tmp/logs/spark.log
sudo mkdir /var/log/cerebralcortex/
sudo chmod 777 /var/log/cerebralcortex/

for filename in ./productionEDDs/daily/*.json; do
    echo $(date) ' starting ' $filename
    python3.6 experiment_engine.py $filename fit
    echo $(date) ' completed ' $filename
done