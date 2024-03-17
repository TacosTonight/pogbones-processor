from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType
from data_frame_processor import DataFrameProcessor
from transformers import pipeline
from sentence_transformers.cross_encoder import CrossEncoder
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
            "1": -1,
            "2": -1,
            "3": -1,
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
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .option("subscribe", os.getenv("KAFKA_TOPIC_NAME"))
        .load()
    )
    decoded_messages_df = initial_messages_df.withColumn(
        "value", initial_messages_df["value"].cast("string")
    )

    # Instantiate hugging face models...
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    )
    cross_encoder_model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

    # Instantiate the dataframe utils...
    df_processor = DataFrameProcessor(
        sentiment_model=sentiment_model, cross_encoder_model=cross_encoder_model
    )

    window_spec = "1 minute"

    # Add sentiment to the messages...
    classified_messages_df = df_processor.add_sentiment(
        dataframe=decoded_messages_df,
    )

    # Get word count of each message
    word_count_df = df_processor.get_word_count(
        classified_messages_df, window_spec, "value"
    )

    # Find most common sentiment...
    most_common_sentiment_df = df_processor.find_most_common_word_in_window(
        classified_messages_df, window_spec, "sentiment"
    )

    # Find the most common word in a window...
    most_common_word_df = df_processor.find_most_common_word_in_window(
        decoded_messages_df, window_spec, "value"
    )

    # Rename the columns...
    most_common_sentiment_df = most_common_sentiment_df.withColumnRenamed(
        "word", "avg_sentiment"
    ).drop("word_count")
    most_common_word_df = most_common_word_df.withColumnRenamed(
        "word", "most_common_word"
    ).drop("word_count")

    # Join the average sentiment and the most common word...
    top_words_and_average_sentiment = most_common_word_df.join(
        most_common_sentiment_df, on="window", how="inner"
    )

    # Add a button to the dataframe...
    final_df = df_processor.add_button(
        dataframe=top_words_and_average_sentiment,
    )

    # Write the results to the console...
    query = (
        final_df.writeStream.outputMode("append")
        .format("console")
        .option("truncate", False)
        .start()
    )
    query.awaitTermination()
