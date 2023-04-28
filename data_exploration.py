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
    peek_at_data(dfs)
    missing_data_exploration(dfs)
  

def get_data(url1, url2):
    gdown.download_folder(url1, quiet=True, use_cookies=False)
    gdown.download_folder(url2, quiet=True, use_cookies=False)

def upload_dfs(paths):
    relevant_paths = [path for path in paths if path.endswith(".txt")] 
    dfs = []
    for i in range(len(relevant_paths)):
        path = relevant_paths[i]
        if path == "kegg_names.txt":
            dfs.append((pd.read_fwf(path, header=None, names=["SampleID", "pathway_name"]), path))
        elif path == "metadata.txt":
            dfs.append((pd.read_table(path, header=0), path))
        else:
            dfs.append((pd.read_csv(path, sep=' ', header=0), path))
        dfs[i][0].set_index("SampleID", inplace=True)
    return dfs

def peek_at_data(dfs_list):
    for elm in dfs_list:
        print(f'no. of samples: %d, no. of {elm[1].split(".")[0]+"_cols"}: %d ' % elm[0].shape)


def missing_data_exploration(dfs_list):
    for elm in dfs_list:
        df = elm[0]
        if ("metadata" in elm[1]):
            total_missing = df.isnull().sum().sum()
        elif ("kegg_names" in elm[1]):
            continue
        else:
            df[df == 0] = None 
            total_missing = np.sum(np.isnan(df.values))
        total_obs = np.prod(df.shape)
        print(f'In {elm[1].split(".")[0]}-')
        print('toal missing values: %d out of total observations: %d' % (total_missing, total_obs))
        print('so that\'s %.2f missing \n' % (total_missing/total_obs))

if __name__ == "__main__":
    main()
    


