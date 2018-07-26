"""
This program allow to create Excel file with the results of the results
file obtained by the CEC'2013 Large Scale Global Optimization file.
"""
import glob
import os
import sys

import itertools

import numpy as np
import pandas as pd
import re

# From https://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n
def mygrouper(n, iterable):
    args = [iter(iterable)] * n
    return ([e for e in t if e is not None] for t in itertools.zip_longest(*args))


def toLatex(global_df, filename_latex):
    """
    Transform the DataFrame to Latex
    """
    def extract(milestone, name, df_grouped, funs):
        bests = df_grouped[funs].tolist()
        values = [milestone, name] + bests
        return values

    def fun2latex(name):
        return re.sub('F0?(\d*)$', '$f_{\\1}$', name)

    def formatter(value):
        return "{:2e}".format(value)

    all_funs = [name for name in global_df.columns if name.startswith('F')]
    # milestones = global_df['milestone'].unique()
    milestones = [1.2e5, 6e5, 3e6]
    total = 5*len(milestones)
    pd.set_option('display.float_format', formatter)

    for funs in mygrouper(4, all_funs):
        df_latex = pd.DataFrame(columns=['Milestone', 'Category']+funs,
                                index=np.arange(total))
        i = 0

        for milestone, df in global_df.groupby(['milestone']):
            if milestone not in milestones:
                continue

            df_latex.loc[i] = extract(milestone, 'Best', df.min(), funs)
            df_latex.loc[i+1] = extract(milestone, 'Median', df.median(), funs)
            df_latex.loc[i+2] = extract(milestone, 'Worst', df.max(), funs)
            df_latex.loc[i+3] = extract(milestone, 'Mean', df.mean(), funs)
            df_latex.loc[i+4] = extract(milestone, 'StDev', df.std(), funs)
            df_latex[df_latex.isnull()] = 0
            i += 5

        df_latex.columns = [fun2latex(col) for col in df_latex.columns]
        df_latex['Milestone'] = df_latex['Milestone'].map('{:.2e}'.format)
        print(df_latex[df_latex['Category']=='Mean'])
        df_latex = df_latex.set_index(['Milestone', 'Category'])
        # print(df_latex[df_latex['Category']=='Best'])

        with open(filename_latex, "a") as file:
            # print(df_latex.groupby(['milestone', 'category']))
            df_latex.to_latex(file, multirow=True, multicolumn=False,
                              escape=False, float_format=formatter)


def changeFun(value):
    if value < 10:
        value = 'F0%s' % value
    else:
        value = 'F%s' % value

    return value


def main(args):
    if len(args) > 1:
        files = args
        if len(args) == 1 and os.path.isfile(args):
            dirname = args[0]
    else:
        if args:
            dirname = args[0]
        else:
            dirname = "."

        files = glob.glob("{}/*.csv".format(dirname))

    dfs = []

    # Concat all .csv files
    for file in files:
        df_file = pd.read_csv(file, sep=",", names=['milestone', 'function',
                                                    'fitness'])
        size, cols = df_file.shape
        maxrun = df_file['milestone'].value_counts().max()

        if (size % maxrun):
            print("Error: not all milestones appears the same number of times")
            sys.exit(1)

        num_mils = int(size/maxrun)
        df_file['run'] = np.repeat(np.arange(1, maxrun+1), num_mils)
        dfs.append(df_file)

    df = pd.concat(dfs, ignore_index=True)
    df.sort_values(['milestone', 'function'])
    df['function'] = df['function'].map(changeFun)
    df_all = pd.pivot_table(df, values='fitness', index=['milestone', 'run'],
                            columns=['function'], aggfunc=np.mean)
    df_all = df_all.reset_index().drop(['run'], axis=1)
    # df_all.index.names = ['index']
    df_all.columns.names = ['']
    df_all.to_excel("results_all.xls", header=True, index=False)
    toLatex(df_all, "results.tex")


if __name__ == '__main__':
    main(sys.argv[1:])
