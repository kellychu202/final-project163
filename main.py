import pandas as pd

def load_merge():
    movies_df = pd.read_csv("/data/Movies_Streaming_Platforms.csv")
    tv_df = pd.read_csv("/data/TvShows_Streaming_Platforms.csv")
    combine_df = [movies_df, tv_df]
    movies_tv = pd.concat(combine_df).reset_index(drop=True)
    print(movies_df)
    pd.merge()

if __name__ == '__main__':  
    main()