import random
from pyspark.sql.functions import (
    window,
    explode,
    split,
    count,
    col,
    udf,
    struct,
    max as max_,
)
from pyspark.sql.types import StringType


class DataFrameProcessor:
    def __init__(self, sentiment_model):
        self.sentiment_model = sentiment_model

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
            dataframe.withWatermark("timestamp", "5 seconds")
            .groupBy(window(col("timestamp"), window_spec), col(column))
            .agg(count("*").alias("count"))
        )

        max_word_count = (
            word_count.groupBy("window")
            .agg(max_(struct(col("count"), col(column))).alias("max_count"))
            .select(
                col("window"),
                col("max_count.{}".format(column)).alias("avg_{}".format(column)),
                col("max_count.count").alias("count"),
            )
        )
        return max_word_count

    def _choose_option(self, sentiment):
        positive_choices = ["up", "right", "A"]
        negative_choices = ["down", "left", "B"]

        # first rank is the most similar, last rank is the least similar
        if sentiment == "positive":
            return random.choice(positive_choices)
        elif sentiment == "negative":
            return random.choice(negative_choices)
        else:
            return random.choice(positive_choices + negative_choices)

    def add_button(self, dataframe):
        choose_option_udf = udf(self._choose_option, StringType())
        return dataframe.withColumn(
            "button",
            choose_option_udf(
                col("avg_sentiment"),
            ),
        )
