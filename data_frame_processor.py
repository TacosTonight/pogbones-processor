import random
from pyspark.sql.functions import (
    window,
    explode,
    split,
    count,
    col,
    row_number,
    udf,
)
from pyspark.sql import Window
from pyspark.sql.types import StringType


class DataFrameProcessor:
    def __init__(self, sentiment_model, cross_encoder_model):
        self.sentiment_model = sentiment_model
        self.cross_encoder_model = cross_encoder_model

    def _classify_message(self, message):
        return self.sentiment_model(message)[0].get("label")

    def add_sentiment(self, dataframe):
        classify_message_udf = udf(self._classify_message, StringType())
        return dataframe.withColumn(
            "sentiment",
            classify_message_udf(col("value")),
        )

    @staticmethod
    def get_word_count_in_window(dataframe, window_spec, column):
        return (
            dataframe.withColumn("word", explode(split(col(column), " ")))
            .groupBy(window(col("timestamp"), window_spec), col("word"))
            .agg(count("*").alias("word_count"))
        )

    @staticmethod
    def find_most_common_word_in_window(dataframe):
        window = Window.partitionBy("window").orderBy(col("word_count").desc())
        return (
            dataframe.withColumn(
                "rank",
                row_number().over(window),
            )
            .filter(col("rank") == 1)
            .select(col("word"), col("window"))
        )

    def _choose_option(self, most_common_word, sentiment):
        query = most_common_word
        corpus = ["L", "R", "down", "up", "left", "right", "start", "select", "B", "A"]
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
