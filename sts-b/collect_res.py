import os
import pandas as pd

datadir = "checkpoint"
files = os.listdir("checkpoint")

dicts = {"models": [], "mae": [], "mse": [], "G_mean": [], "Pearson": [], "Spearman": [],
         "Lmae": [], "Lmse": [], "LG_mean": [], "LPearson": [], "LSpearman": [],
         "Mmae": [], "Mmse": [], "MG_mean": [], "MPearson": [], "MSpearman": [],
         "Smae": [], "Smse": [], "SG_mean": [], "SPearson": [], "SSpearman": []}
for i in files:
    # i="rank_best-param_mse_seed_42_valint_400_patience_10_adam_0.00025_16"
    with open(os.path.join(datadir, i, "training.log")) as f:
        strs = f.readlines()

    """
    ***** TEST RESULTS *****
    2023-12-20 20:05:27,986 |  * Overall: MSE 1.162	L1 0.875	G-Mean 0.579	Pearson 0.670	Spearman 0.668	Number 1000
    2023-12-20 20:05:27,986 |  * Many: MSE 1.103	L1 0.848	G-Mean 0.564	Pearson 0.637	Spearman 0.612	Number 756
    2023-12-20 20:05:27,986 |  * Medium: MSE 1.359	L1 0.975	G-Mean 0.652	Pearson 0.621	Spearman 0.470	Number 170
    2023-12-20 20:05:27,986 |  * Few: MSE 1.316	L1 0.917	G-Mean 0.569	Pearson 0.644	Spearman 0.613	Number 74
    2023-12-20 20:05:27,989 | Done testing.
    """
    try:
        few = strs[-2][26:].replace('\t', " ").split(" ")
        Medium = strs[-3][26:].replace('\t', " ").split(" ")
        Many = strs[-4][26:].replace('\t', " ").split(" ")
        Overall = strs[-5][26:].replace('\t', " ").split(" ")

        print(few)
        # print(strs[-3][26:].split(" "))
        # print(strs[-4][26:].split(" "))
        # print(strs[-5][26:].split(" "))
        # mse = float(test_str[3][1:-2])
        # mae = float(test_str[5][1:-2])
        # G_mean = float(test_str[-1][1:-2])
        # test_str2 = strs[-3][26:].split(" ")
        # r2 = float(test_str2[-1][:-1])

        # print(mae, mse, G_mean, r2)
        # items = [i, mae, mse, G_mean, r2]

        dicts['mae'].append(float(Overall[6]))
        dicts['mse'].append(float(Overall[4]))
        dicts['G_mean'].append(float(Overall[8]))
        dicts['Pearson'].append(float(Overall[10]))
        dicts['Spearman'].append(float(Overall[12]))
        # print(dicts)

        dicts['Lmae'].append(float(Many[6]))
        dicts['Lmse'].append(float(Many[4]))
        dicts['LG_mean'].append(float(Many[8]))
        dicts['LPearson'].append(float(Many[10]))
        dicts['LSpearman'].append(float(Many[12]))

        dicts['Mmae'].append(float(Medium[6]))
        dicts['Mmse'].append(float(Medium[4]))
        dicts['MG_mean'].append(float(Medium[8]))
        dicts['MPearson'].append(float(Medium[10]))
        dicts['MSpearman'].append(float(Medium[12]))

        dicts['Smae'].append(float(few[6]))
        dicts['Smse'].append(float(few[4]))
        dicts['SG_mean'].append(float(few[8]))
        dicts['SPearson'].append(float(few[10]))
        dicts['SSpearman'].append(float(few[12]))
        dicts['models'].append(i)
    except:
        for k in dicts:
            if k!='models':
                dicts[k].append('nan')
        dicts['models'].append(i)
        continue
print(dicts)
df = pd.DataFrame(dicts)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--o', default="./ablition_res2.xlsx", help='save dictionary')
    parser.set_defaults(augment=True)
    args, unknown = parser.parse_known_args()
    df.to_excel(args.o)