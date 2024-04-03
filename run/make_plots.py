import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # <--- Add this line
import matplotlib.pyplot as plt
import argparse
import json
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='toy')
    args = parser.parse_args()

    # Load data.
    with open(f'../collected/{args.dataset}_results.json', 'r') as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    df_tmp = df.melt(id_vars=['alpha', 'trials'], var_name='indicator', value_name='score')
    
    try: 
        path = os.path.join("../collected", f"{args.dataset}")
        os.makedirs(path, exist_ok = True) 
    except OSError as error: 
        print("Directory '%s' can not be created" % path) 
    for var in ['inefficiencies', 'true_coverages', 'top1_coverages', 'aggregated_coverage']:
        # select rows whose 'indicator' column contains var
        df_plot = df_tmp[df_tmp['indicator'].str.contains(var)]
        plt.figure()
        sns.boxplot(x="alpha",
                    y="score",
                    hue="indicator",
                    data=df_plot,
                    width=0.6,
                    linewidth=0.6,
                    showmeans=False,
                    fliersize=1,
                    )  
        plt.savefig(f'{path}/{args.dataset}_{var}.pdf')
