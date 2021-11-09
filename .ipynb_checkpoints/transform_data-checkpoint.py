import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd

# Transform data directories with classes to a dataframe....
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transform the directory of .npz files to a single dataframe.')
    parser.add_argument('data_dir', type=str, help='Root data directory to read (subdirectories will be considered classes)')
    parser.add_argument('output', type=str, help='Name of the output file')
    args = parser.parse_args()

    # List all files and directories
    files = glob.glob(os.path.join(args.data_dir, "*/*.npz"))
    histograms = []
    colunm_names = ["name", "class"] + [str(i) for i in range(65)]
    for f in files:
        # Slit class and filename
        classe, filename = f.split('/')[-2], f.split('/')[-1]
        classe = int(classe)
        name = filename[:-4]
        # Read histogram
        histogram = np.load(f)["values"]
        sample = [name, classe] + list(histogram)
        histograms.append(sample)

    # Generate a dataframe
    df = pd.DataFrame(histograms, columns=colunm_names)
    df.to_csv(args.output)

