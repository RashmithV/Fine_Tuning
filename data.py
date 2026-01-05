import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_parquet("hf://datasets/bharatgenai/BhashaBench-Finance/English/test-00000-of-00001.parquet")
df.to_csv("data.csv", index=False)