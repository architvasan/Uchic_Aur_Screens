import pandas as pd
import numpy as np
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dataset",
    type=Path,
    required=True,
    help="docking training data",
)

parser.add_argument(
    "-o", "--output",
    type=str,
    required=True,
    help="output pattern",
    )
args = parser.parse_args()

df = pd.read_csv(args.dataset)
df['scores'] = -df['scores']
df = df[df['scores'] > -5]
df['scores'] = df['scores'].apply(lambda x: max(x, 0))
#df[df['scores']<0] = 0
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
train.to_csv(f"{args.output}.train", index=False)
test.to_csv(f"{args.output}.val", index=False)

