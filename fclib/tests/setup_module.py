# import pandas as pd
# import numpy as np
# from itertools import product

# def setup_module(module):
#     keyvars = {
#         "store": [1, 2],
#         "brand": [1, 2, 3],
#         "week": list(range(50, 61))
#     }
#     df = pd.DataFrame([row for row in product(*keyvars.values())], 
#                       columns=keyvars.keys())

#     n = len(df)
#     np.random.seed(12345)
#     df["constant"] = 1
#     df["logmove"] = np.random.normal(9, 1, n)
#     df["price1"] = np.random.normal(0.55, 0.003, n)
#     df["price2"] = np.random.normal(0.55, 0.003, n)
#     df["price3"] = np.random.normal(0.55, 0.003, n)
#     df["price4"] = np.random.normal(0.55, 0.003, n)
#     df["price5"] = np.random.normal(0.55, 0.003, n)
#     df["price6"] = np.random.normal(0.55, 0.003, n)
#     df["price7"] = np.random.normal(0.55, 0.003, n)
#     df["price8"] = np.random.normal(0.55, 0.003, n)
#     df["price9"] = np.random.normal(0.55, 0.003, n)
#     df["price10"] = np.random.normal(0.55, 0.003, n)
#     df["price11"] = np.random.normal(0.55, 0.003, n)
#     df["deal"] = np.random.binomial(1, 0.5, n)
#     df["feat"] = np.random.binomial(1, 0.25, n)
#     df["profit"] = np.random.normal(30, 7.5, n)
#     df.to_csv("tests/resources/ojdatasim.csv")
