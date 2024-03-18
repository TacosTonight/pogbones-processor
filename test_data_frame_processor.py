import unittest
import warnings
from unittest.mock import patch
from data_frame_processor import DataFrameProcessor
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType


class TestDataFrameProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", category=ResourceWarning)
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("DataFrameProcessorTest")
            .getOrCreate()
        )
        cls.dataframe_processor = DataFrameProcessor(None)

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

    def test_find_most_common_word_in_window(self):
        """
        SCENARIO
            - Find most common word in a DataFrame
        EXPECTED RESULT
            - The DataFrame should be one row with the most common word as positive
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
                "positive",
                "2024-02-17 18:31:00.767",
                "channel1",
            ),
            (
                "2",
                "positive",
                "2024-02-17 18:31:00.767",
                "channel1",
            ),
            (
                "2",
                "negative",
                "2024-02-17 18:31:00.767",
                "channel1",
            ),
        ]

        # Create a DataFrame
        initial_messages_df = self.spark.createDataFrame(data, schema)
        result = self.dataframe_processor.find_most_common_word_in_window(
            initial_messages_df, "10 seconds", "value"
        )
        self.assertEqual(result.columns, ["window", "avg_value", "count"])
        self.assertEqual(result.count(), 1)
        self.assertEqual(result.collect()[0]["avg_value"], "positive")

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
                StructField("avg_sentiment", StringType(), False),
            ]
        )

        # Define the data as a list of tuples
        data = [
            (
                "windowspec",
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
                ["window", "avg_sentiment", "button"],
            )
            self.assertEqual(result.count(), 1)

    def test_choose_option_positive(self):
        """
        SCENARIO
            - Choose an option based on the sentiment
        EXPECTED RESULT
            - The option should be either "up", "right", or "A"
        """
        result = self.dataframe_processor._choose_option("positive")
        self.assertIn(result, ["up", "right", "A"])

    def test_choose_option_negative(self):
        """
        SCENARIO
            - Choose an option based on negative sentiment
        EXPECTED RESULT
            - The option should be either "down", "left", or "B"
        """
        result = self.dataframe_processor._choose_option("negative")
        self.assertIn(result, ["down", "left", "B"])
