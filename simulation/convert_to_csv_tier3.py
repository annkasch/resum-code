import numpy as np
import pandas as pd
from NeutronSimulation import NeutronSimulation as sim
import argparse
import os
import sys
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
resum_path = os.getenv("RESUM_PATH")
if resum_path is None:
    raise ValueError("Environment variable RESUM_PATH is not set. Make sure to define it in your .env file.")
utilities_path = os.path.join(resum_path, "utilities")
sys.path.append(utilities_path)
import utilities as utils


def main(filename, path_to_files, fileout): # python file

    filename = filename.replace("root", "csv/tier2")
    filename = filename.replace("tier0", "tier2")
    filename = filename.replace("tier1", "tier2")
    
    index = filename.find("-tier2")
    filename = filename[:index + len("-tier2")]

        
    if path_to_files != "":
        filename='{}/{}'.format(path_to_files,filename)

    df_new,_ = utils.get_dataframes_concat(filename, df_new=pd.DataFrame())

    
    if fileout=="":
        fileout = filename.replace("tier2", "tier3")
        fileout = fileout+".csv"

    index = fileout.find("/tier3/")
    path_out = fileout[:index + len("/tier3")]

    Path('{}'.format(path_out)).mkdir(parents=True, exist_ok=True)
    df_new.to_csv(fileout)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',type=str, default=["neutron-sim-LF"],help="List of filenames to process")
    parser.add_argument('--path_to_files', type=str, default="")
    parser.add_argument('--filename_out', type=str, default="")
    args = parser.parse_args()

    main(args.filename, args.path_to_files, args.filename_out)