import unittest
from unittest.mock import patch, MagicMock
from data_frame_processor import DataFrameProcessor
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType


class TestDataFrameProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("DataFrameProcessorTest")
            .getOrCreate()
        )
        # Mock the cross encoder model
        cls.cross_encoder_model_mock = MagicMock()
        cls.dataframe_processor = DataFrameProcessor(None, cls.cross_encoder_model_mock)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_add_sentiment(self):
        """
        SCENARIO
            - Add sentiment to a DataFrame
        EXPECTED RESULT
            - The DataFrame should have a new column called "sentiment" with the sentiment of the message
        """
        # Define the schema for the DataFrame
        schema = StructType(
            [
                StructField("key", StringType(), False),
                StructField("value", StringType(), False),
                StructField(
                    "timestamp", StringType(), False
                ),  # Assuming the timestamp is a string
                StructField("channel", StringType(), False),
            ]
        )

        # Define the data as a list of tuples
        data = [
            (
                "1",
                "Hey everyone, how's it going?",
                "2024-02-17 18:31:00.767",
                "channel1",
            ),
        ]

        # Create a DataFrame
        initial_messages_df = self.spark.createDataFrame(data, schema)

        # Mock _classify_message method
        with patch.object(
            self.dataframe_processor, "_classify_message", return_value="positive"
        ):
            result = self.dataframe_processor.add_sentiment(initial_messages_df)
            self.assertEqual(
                result.columns, ["key", "value", "timestamp", "channel", "sentiment"]
            )
            self.assertEqual(result.count(), 1)

    def test_get_word_count_in_window(self):
        """
        SCENARIO
            - Count the words in a DataFrame
        EXPECTED RESULT
            - The DataFrame should have a new column called "word_count" with the count of each word
        """
        # Define the schema for the DataFrame
        schema = StructType(
            [
                StructField("key", StringType(), False),
                StructField("value", StringType(), False),
                StructField(
                    "timestamp", StringType(), False
                ),  # Assuming the timestamp is a string
                StructField("channel", StringType(), False),
            ]
        )

        # Define the data as a list of tuples
        data = [
            (
                "1",
                "Hi Hey Hello",
                "2024-02-17 18:31:00.767",
                "channel1",
            ),
        ]

        # Create a DataFrame
        initial_messages_df = self.spark.createDataFrame(data, schema)
        result = self.dataframe_processor.get_word_count_in_window(
            initial_messages_df, "10 seconds", "value"
        )
        self.assertEqual(result.columns, ["window", "word", "word_count"])
        self.assertEqual(result.count(), 3)

    def test_find_most_common_word_in_window(self):
        """
        SCENARIO
            - Find the most common word in a DataFrame
        EXPECTED RESULT
            - The DataFrame should have a new column called "rank" with the rank of each word
        """
        # Define the schema for the DataFrame
        schema = StructType(
            [
                StructField("window", StringType(), False),
                StructField("word", StringType(), False),
                StructField("word_count", StringType(), False),
            ]
        )

        # Define the data as a list of tuples
        data = [
            (
                "2024-02-17 18:31:00.767",
                "Hi",
                "1",
            ),
            (
                "2024-02-17 18:31:00.767",
                "Hey",
                "2",
            ),
            (
                "2024-02-17 18:31:00.767",
                "Hello",
                "3",
            ),
        ]

        # Create a DataFrame
        initial_messages_df = self.spark.createDataFrame(data, schema)
        result = self.dataframe_processor.find_most_common_word_in_window(
            initial_messages_df
        )
        self.assertEqual(result.columns, ["word", "window"])
        self.assertEqual(result.count(), 1)

    def test_add_button(self):
        """
        SCENARIO
            - Add a button to a DataFrame
        EXPECTED RESULT
            - The DataFrame should have a new column called "button" with the button to be pressed
        """
        # Define the schema for the DataFrame
        schema = StructType(
            [
                StructField("window", StringType(), False),
                StructField("most_common_word", StringType(), False),
                StructField("avg_sentiment", StringType(), False),
            ]
        )

        # Define the data as a list of tuples
        data = [
            (
                "windowspec",
                "Hi",
                "positive",
            ),
        ]

        # Create a DataFrame
        initial_messages_df = self.spark.createDataFrame(data, schema)

        # Mock _choose_option method
        with patch.object(self.dataframe_processor, "_choose_option", return_value="L"):
            result = self.dataframe_processor.add_button(initial_messages_df)
            self.assertEqual(
                result.columns,
                ["window", "most_common_word", "avg_sentiment", "button"],
            )
            self.assertEqual(result.count(), 1)

    def test_choose_option_positive(self):
        """
        SCENARIO
            - Choose an option based on the most common word and positive sentiment
        EXPECTED RESULT
            - The option should be "L"
        """
        self.cross_encoder_model_mock.rank.return_value = [
            {"corpus_id": 0, "score": 0.37},
            {"corpus_id": 1, "score": 0.34},
            {"corpus_id": 2, "score": 0.32},
            {"corpus_id": 3, "score": 0.30},
            {"corpus_id": 4, "score": 0.28},
            {"corpus_id": 5, "score": 0.27},
            {"corpus_id": 6, "score": 0.26},
            {"corpus_id": 7, "score": 0.25},
            {"corpus_id": 8, "score": 0.25},
            {"corpus_id": 9, "score": 0.23},
        ]

        most_common_word = "Hi"
        sentiment = "positive"
        option = self.dataframe_processor._choose_option(most_common_word, sentiment)
        self.assertEqual(option, "L")

    def test_choose_option_negative(self):
        """
        SCENARIO
            - Choose an option based on the most common word and negative sentiment
        EXPECTED RESULT
            - The option should be "A"
        """
        self.cross_encoder_model_mock.rank.return_value = [
            {"corpus_id": 0, "score": 0.37},
            {"corpus_id": 1, "score": 0.34},
            {"corpus_id": 2, "score": 0.32},
            {"corpus_id": 3, "score": 0.30},
            {"corpus_id": 4, "score": 0.28},
            {"corpus_id": 5, "score": 0.27},
            {"corpus_id": 6, "score": 0.26},
            {"corpus_id": 7, "score": 0.25},
            {"corpus_id": 8, "score": 0.25},
            {"corpus_id": 9, "score": 0.23},
        ]

        most_common_word = "Hi"
        sentiment = "negative"
        option = self.dataframe_processor._choose_option(most_common_word, sentiment)
        self.assertEqual(option, "A")
