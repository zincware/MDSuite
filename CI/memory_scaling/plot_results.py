import argparse
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("db")
    parser.add_argument("plot")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = get_parser()
    con = sqlite3.connect(parsed.db)
    df = pd.read_sql_query("SELECT * FROM TEST_METRICS", con)

    # get the name of the test without the number of points
    df["test"] = df["ITEM_VARIANT"].str.split("[").str[0]
    # set the index to the number of points
    df = df.set_index("ITEM_VARIANT")
    df.index = df.index.str.extract(r"(\d+)", expand=False)

    # group by memory usage
    grouped = df[["MEM_USAGE", "test"]].groupby("test")

    ncols = 2
    nrows = int(np.ceil(grouped.ngroups / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
        grouped.get_group(key).plot(ax=ax)
        ax.legend()
        ax.set_title(key)

    fig.savefig(parsed.plot)
    print(
        df[
            ["KERNEL_TIME", "CPU_USAGE", "MEM_USAGE", "test", "TOTAL_TIME", "USER_TIME"]
        ].to_markdown()
    )
