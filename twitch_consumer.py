from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, LongType
from data_frame_processor import DataFrameProcessor
from transformers import pipeline
from sentence_transformers.cross_encoder import CrossEncoder

if __name__ == "__main__":
    # Load data from Kafka...
    spark = (
        SparkSession.builder.appName("SimplePySparkExample")
        .config("spark.executor.memory", "3g")
        .config("spark.driver.memory", "3g")
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

    initial_messages_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", "host1:port1,host2:port2")
        .option("subscribe", "topic1")
        .schema(twitch_message_schema)
        .load()
    )

    # Convert the timestamp to a human-readable format...
    initial_messages_df = initial_messages_df.withColumn(
        "timestamp", (col("timestamp") / 1000).cast("timestamp")
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

    # Add sentiment to the messages...
    classified_messages_df = df_processor.add_sentiment(
        dataframe=initial_messages_df,
    )

    # Count Sentiments...
    sentiment_count_df = df_processor.get_word_count_in_window(
        classified_messages_df, "1 minute", "sentiment"
    )

    # Find most common sentiment...
    most_common_sentiment_df = df_processor.find_most_common_word_in_window(
        sentiment_count_df
    )

    # Calculate the word count in a window...
    word_count_df = df_processor.get_word_count_in_window(
        dataframe=initial_messages_df, window_spec="1 minute", column="value"
    )

    # Find the most common word in a window...
    most_common_word_df = df_processor.find_most_common_word_in_window(
        dataframe=word_count_df
    )

    # Rename the columns...
    most_common_sentiment_df = most_common_sentiment_df.withColumnRenamed(
        "word", "avg_sentiment"
    )
    most_common_word_df = most_common_word_df.withColumnRenamed(
        "word", "most_common_word"
    )

    # Join the average sentiment and the most common word...
    top_words_and_average_sentiment = most_common_word_df.join(
        most_common_sentiment_df, on="window", how="inner"
    )

    top_words_and_average_sentiment.show()

    # Add a button to the dataframe...
    final_df = df_processor.add_button(
        dataframe=top_words_and_average_sentiment,
    )
