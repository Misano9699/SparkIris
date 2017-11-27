# SparkIris

An [Apache Spark Machine Learning](https://spark.apache.org/mllib/) example for predicting flower species 
from the classic Iris dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Iris).
It contains three iris species with 50 samples each. 
 
## How to run on a standalone cluster

The code runs on a standalone Apache Spark cluster. 

The steps to take are:
1. Build the jar
2. Copy the jar and dataset to a location on the Apache Spark cluster
3. Submit the jar to Apache Spark

For more information on how to install a standalone Spark cluster see <https://spark.apache.org/docs/latest/spark-standalone.html>

For submitting the application see <https://spark.apache.org/docs/latest/submitting-applications.html>

## Example 

I used a Spark Docker image from Semantive which can be found on [Github](https://github.com/Semantive/docker-spark)

**Usage:**

* Clone the *docker-spark* repository to your local machine
* Adapt *docker-compose.yml* to your liking (for example, number of workers, number of cores, memory allocated to the workers)
* Run *docker-compose up*, which will start the Spark cluster
* Copy both the Iris dataset and the SparkIris jar to the *data* directory of the docker-spark repository
* Connect to the master image with a bash shell (*docker exec -i -t dockerspark_master_1 /bin/bash*)
* In the bash shell submit the application to Spark (*./bin/spark-submit --class nl.craftsmen.spark.iris.SparkIris --master local[1] /tmp/data/spark-iris-1.0-SNAPSHOT.jar*)


