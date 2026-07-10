import pandas as pd
for f in ["dataset03", "dataset04"]:
    df = pd.read_csv(f"data/raw/batadal/BATADAL_{f}.csv")
    df.columns = df.columns.str.strip()          # enlève les espaces parasites
    print(f, "| shape:", df.shape, "| ATT_FLAG:", df["ATT_FLAG"].value_counts().to_dict())

