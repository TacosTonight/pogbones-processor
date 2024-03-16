import random
from pyspark.sql.functions import (
    window,
    explode,
    split,
    count,
    col,
    row_number,
    udf,
    struct,
    max as max_,
)
from pyspark.sql import Window
from pyspark.sql.types import StringType


class DataFrameProcessor:
    def __init__(self, sentiment_model, cross_encoder_model):
        self.sentiment_model = sentiment_model
        self.cross_encoder_model = cross_encoder_model

    def _classify_message(self, message):
        if self.sentiment_model is None:
            return random.choice(["positive", "negative", "neutral"])
        return self.sentiment_model(message)[0].get("label")

    def add_sentiment(self, dataframe):
        classify_message_udf = udf(self._classify_message, StringType())
        return dataframe.withColumn(
            "sentiment",
            classify_message_udf(col("value")),
        )

    @staticmethod
    def find_most_common_word_in_window(dataframe, window_spec, column):
        word_count = (
            dataframe.withColumn("word", explode(split(col(column), " ")))
            .withWatermark("timestamp", "5 seconds")
            .groupBy(window(col("timestamp"), window_spec), col("word"))
            .agg(count("*").alias("word_count"))
        )
        max_word_count = (
            word_count.groupBy("window")
            .agg(max_(struct(col("word_count"), col("word"))).alias("max_word_count"))
            .select(
                col("window"),
                col("max_word_count.word").alias("word"),
                col("max_word_count.word_count").alias("word_count"),
            )
        )
        return max_word_count

    def _choose_option(self, most_common_word, sentiment):
        query = most_common_word
        corpus = ["L", "R", "down", "up", "left", "right", "start", "select", "B", "A"]
        if self.cross_encoder_model is None:
            return random.choice(corpus)
        ranks = self.cross_encoder_model.rank(query, corpus)

        # first rank is the most similar, last rank is the least similar
        if sentiment == "positive":
            return corpus[ranks[0].get("corpus_id")]
        elif sentiment == "negative":
            return corpus[ranks[-1].get("corpus_id")]
        else:
            return random.choice(corpus)

    def add_button(self, dataframe):
        choose_option_udf = udf(self._choose_option, StringType())
        return dataframe.withColumn(
            "button",
            choose_option_udf(
                col("most_common_word"),
                col("avg_sentiment"),
            ),
        )
