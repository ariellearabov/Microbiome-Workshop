import pandas as pd
import numpy as np
import gdown
import os

def main():
    omic = "https://drive.google.com/drive/folders/1_GzgqZ2tVaUzUvaFGwipjysiOEUgCZDS?usp=share_link"
    meta = "https://drive.google.com/drive/folders/1YU84ZOk_BNGIOxNrATO_VhtLkHSlv-w3?usp=share_link"
    #get_data(omic, meta)  # doesn't work since python has no premission to acsess the folders so we downloaded them manually. 
    
    os.chdir("data")
    paths = os.listdir()
    dfs = upload_dfs(paths)
    print(dfs[0][0])
    

def get_data(url1, url2):
    gdown.download_folder(url1, quiet=True, use_cookies=False)
    gdown.download_folder(url2, quiet=True, use_cookies=False)

def upload_dfs(paths):
    relevant_paths = [path for path in paths if path.endswith(".txt")] 
    dfs = []
    for i in range(len(relevant_paths)):
        path = relevant_paths[i]
        if path == "kegg_names.txt":
            dfs.append((pd.read_fwf(path, header=None, names=["ID", "pathway_name"]), path))
        elif path == "metadata.txt":
            dfs.append((pd.read_table(path, header=0), path))
        else:
            dfs.append((pd.read_csv(path, sep=' ', header=0), path))
    return dfs


if __name__ == "__main__":
    main()
    


