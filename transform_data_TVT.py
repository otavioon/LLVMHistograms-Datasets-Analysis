import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import yaml

def  generate_ds_file(ds_set, data_dir, output_filename):
    print("Generating file", output_filename)
    dataset_set = []
    for ds_class, ds_progs in ds_set.items():
        dataset_set.extend(ds_progs)
    files = glob.glob(os.path.join(data_dir, "*/*.npz"))
    histograms = []
    colunm_names = ["name", "class"] + [str(i) for i in range(65)]
    for f in files:
        # Slit class and filename
        classe, filename = f.split('/')[-2], f.split('/')[-1]
        classe = int(classe)
        name = filename[:-4]
        # Check whether or not the name belongs to the dataset
        if name in dataset_set:
            # Read histogram
            histogram = np.load(f)["values"]
            sample = [name, classe] + list(histogram)
            histograms.append(sample)
    # Generate a dataframe
    df = pd.DataFrame(histograms, columns=colunm_names)
    df.to_csv(output_filename)

# Transform data directories with classes to a dataframe....
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transform the directory of .npz files to a single dataframe.')
    parser.add_argument('data_dir', type=str, help='Root data directory to read (subdirectories will be considered classes)')
    parser.add_argument('cfg', type=str, help='Name of the configuration file')
    parser.add_argument('output', type=str, help='Name of the output file')
    args = parser.parse_args()


    with open(args.cfg) as file:
        dataset_cfg = yaml.load(file, Loader=yaml.FullLoader)

    for ds_name, ds_sets in dataset_cfg.items():
        print("Dataset name:", ds_name)
        for ds_set_name, ds_set in ds_sets.items():
            # Generate a file with suffix ds_set_name + .csv using ds_set files from data_dir
            print("ds_set_name =", ds_set_name)
            generate_ds_file(ds_set,args.data_dir, args.output + "." + ds_set_name + ".csv")



