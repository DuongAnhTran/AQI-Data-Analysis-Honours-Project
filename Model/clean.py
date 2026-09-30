import pandas as pd
import numpy as np

def clean_data(filename):
    df = pd.read_parquet(filename).copy()
    df_gafanha = df[df['station'].str.lower() == 'p1.gafanha'].copy()
    print(f'gafanha has total {len(df_gafanha)} rows')


    df_gafanha = df_gafanha[df_gafanha['value'].round(2) >= -0.05]
    # df_gafanha.drop(df_gafanha['value'].round(2) < -0.05, inplace=True)
    df_gafanha['values'] = df_gafanha['value'].clip(lower=0)

    print(f'gafanha has total {len(df_gafanha)} rows after cleaning')
    df_gafanha.to_parquet("df_gafanha.parquet")

if __name__ == "__main__":
    clean_data("df_resolved.parquet")
