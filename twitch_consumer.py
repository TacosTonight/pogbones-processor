from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType
from data_frame_processor import DataFrameProcessor
from dotenv import load_dotenv

import os
import json

os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell"
)


if __name__ == "__main__":
    load_dotenv()
    # Load data from Kafka...
    spark = (
        SparkSession.builder.appName("SimplePySparkExample")
        .config("spark.executor.memory", "3g")
        .config("spark.driver.memory", "3g")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    twitch_message_schema = StructType(
        [
            StructField("key", StringType(), True),
            StructField("value", StringType(), True),
            StructField("timestamp", LongType(), True),
            StructField("channel", StringType(), True),
        ]
    )

    starting_offset = {
        os.getenv("KAFKA_TOPIC_NAME"): {
            "0": -1,
            "1": -2,
            "2": -2,
            "3": -2,
            "4": -1,
            "5": -1,
        }
    }

    initial_messages_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", os.getenv("KAFKA_BOOTSTRAP_SERVER"))
        .option("kafka.security.protocol", "SASL_SSL")
        .option("kafka.ssl.endpoint.identification.algorithm", "https")
        .option(
            "kafka.sasl.jaas.config",
            "org.apache.kafka.common.security.plain.PlainLoginModule required username='{}' password='{}';".format(
                os.getenv("KAFKA_KEY"), os.getenv("KAFKA_SECRET")
            ),
        )
        .option("kafka.sasl.mechanism", "PLAIN")
        .option("startingOffsets", json.dumps(starting_offset))
        .option("failOnDataLoss", "false")
        .option("subscribe", os.getenv("KAFKA_TOPIC_NAME"))
        .load()
    )
    decoded_messages_df = initial_messages_df.withColumn(
        "value", initial_messages_df["value"].cast("string")
    )

    # Instantiate the dataframe utils...
    df_processor = DataFrameProcessor(None)

    window_spec = "1 minute"

    # Add sentiment to the messages...
    classified_messages_df = df_processor.add_sentiment(
        dataframe=decoded_messages_df,
    )

    # Find most common sentiment...
    most_common_sentiment_df = df_processor.find_most_common_word_in_window(
        classified_messages_df, window_spec, "sentiment"
    )

    # Add a button to the dataframe...
    final_df = df_processor.add_button(most_common_sentiment_df)

    # Write the results to the console...
    query = (
        final_df.writeStream.outputMode("append")
        .format("console")
        .option("truncate", False)
        .start()
    )
    query.awaitTermination()
