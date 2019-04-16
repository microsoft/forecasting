# Define benchmark related parameters. The parameters should conform with the benchmark definition
# in ../README.md file
 
import os
import pandas as pd
from retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_paths import TRAIN_DIR

NUM_ROUNDS = 12
PRED_HORIZON = 3
PRED_STEPS = 2
TRAIN_START_WEEK = 40
TRAIN_END_WEEK_LIST = list(range(135,159,2))
TEST_START_WEEK_LIST = list(range(137,161,2))
TEST_END_WEEK_LIST = list(range(138,162,2))
# The start datetime of the first week in the record
FIRST_WEEK_START = pd.to_datetime('1989-09-14 00:00:00')

# Get unique stores and brands
train_df = pd.read_csv(os.path.join(TRAIN_DIR, "train_round_1.csv"))
store_list = train_df["store"].unique()
brand_list = train_df["brand"].unique()