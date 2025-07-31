import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# -----------------------------------------------------------------------------
def load_groups_from_csv(csv_path, group_col='id', data_col='name', min_large=20, min_small=1):
    df = pd.read_csv(csv_path)
    groups = df.groupby(group_col)[data_col].apply(list)
    large_groups = [group for group in groups if len(group) >= min_large]
    small_groups = [group for group in groups if min_small <= len(group) < min_large]
    return large_groups, small_groups


# =============================================================================
# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="analyze_wikidata_extract", description="Performs a categorical frequency count on the specified column."
    )

    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file.')
    parser.add_argument('--group_col', type=str, required=False, default='id', help='Column header in CSV file that denotes the group. Default \'id\'')
    parser.add_argument('--data_col', type=str, required=False, default='name', help='Column header in CSV file that denotes the data. Default \'name\'')
    parser.add_argument('--min_large', type=int, required=False, default=20, help='Defines the minium size of a \'large\' group. Default 20')
    parser.add_argument('--min_small', type=int, required=False, default=1, help='Defines the minium size of a \'small\' group. Default 1')
    # parser.add_argument("-o", "--outfile", action="store", required=True)
    # parser.add_argument("infile", type=str, help="File of lines to phrase")
    args = parser.parse_args()

    csv_path = args.csv_path
    base_path, _ = os.path.splitext(csv_path)

    # outfile_path = args.outfile
    # line_count = 0
    # print("☺", flush=True, end='\r')
    # df = pd.read_csv(infile_path)
    # category_counts = df[column].value_counts()
    # if (threshold):
    #     categories_to_keep = category_counts[category_counts >= threshold].index
    #     df_trimmed = df[df[column].isin(categories_to_keep)]
    #     df_trimmed.to_csv(base_path + '-trimmed.csv', index=False)
    #     start_unique_count = df[column].nunique()
    #     trimmed_unique_count = df_trimmed[column].nunique()
    #     print("Number of unique " + column + f"s: {start_unique_count} --> {trimmed_unique_count}", flush=True)

    large_groups, small_groups = load_groups_from_csv(csv_path, group_col=args.group_col, data_col=args.data_col, min_large=args.min_large, min_small=args.min_small)
    print(f"{len(large_groups):>10} Large groups")
    print(f"{len(small_groups):>10} Small groups")
    print(f"{len(large_groups) + len(small_groups):>10} Total groups")

    # # Using Pandas value_counts() and plot()
    # category_counts.plot(kind='bar')
    # plt.title('Frequency of ' + column + ' (Pandas)')
    # plt.xlabel(column)
    # plt.ylabel('count')
    # plt.savefig(base_path + '-pandas.png')
    # # plt.show()

    # # Using Seaborn countplot()
    # sns.countplot(x=column, data=df)
    # plt.title('Frequency of ' + column + ' (Seaborn)')
    # plt.xlabel(column)
    # plt.ylabel('count')
    # plt.savefig(base_path + '-seaborn.png')
    # # plt.show()
