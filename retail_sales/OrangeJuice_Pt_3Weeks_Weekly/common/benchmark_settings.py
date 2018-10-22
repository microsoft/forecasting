# Define benchmark related parameters. The parameters should conform with the benchmark definition
# in ../README.md file
 
import numpy as np

NUM_ROUNDS = 12
TRAIN_START_WEEK = 40
TRAIN_END_WEEK_LIST = np.linspace(135, 157, NUM_ROUNDS, dtype=int)
TEST_START_WEEK_LIST = np.linspace(137, 159, NUM_ROUNDS, dtype=int)
TEST_END_WEEK_LIST = np.linspace(138, 160, NUM_ROUNDS, dtype=int)
