import os
import pandas as pd

datadir = "checkpoint"
files = os.listdir("checkpoint")

dicts = {"models": [], "mae": [], "mse": [], "G_mean": [],
         "Lmae": [], "Lmse": [], "LG_mean": [],
         "Mmae": [], "Mmse": [], "MG_mean": [],
         "Smae": [], "Smse": [], "SG_mean": [],}

for i in files:
    i = "balanced_agedb_resnet50_mse_adam_l1_0.00025_64"
    with open(os.path.join(datadir, i, "training.log")) as f:
        strs = f.readlines()

    """
    2023-12-25 20:57:23,943 |  * Overall: MSE 316.895	L1 14.513	G-Mean 9.779
    2023-12-25 20:57:23,943 |  * Many: MSE 207.987	 RMSE 2.527	L1 11.403	G-Mean 7.345
    2023-12-25 20:57:23,943 |  * Median: MSE 496.941	 RMSE 4.192	L1 20.289	G-Mean 18.054
    2023-12-25 20:57:23,943 |  * Low: MSE 847.561	 RMSE 10.966	L1 27.908	G-Mean 26.781
    2023-12-25 20:57:23,944 | Test loss: MSE [316.8948], L1 [14.5128], G-Mean [9.7792]
    Done
    """
    try:
        test_str = strs[-2][26:].split(" ")
        print(test_str)
        mse = float(test_str[3][1:-2])
        mae = float(test_str[5][1:-2])
        G_mean = float(test_str[-1][1:-2])

        few = strs[-3][26:].replace("\t", " ").split(" ")
        media = strs[-4][26:].replace("\t", " ").split(" ")
        many = strs[-5][26:].replace("\t", " ").split(" ")
        # print(few)
        # print(float(few[9]), float(few[4]), float(few[11][:-1]))

        dicts['mae'].append(mae)
        dicts['mse'].append(mse)
        dicts['G_mean'].append(G_mean)

        dicts['Smae'].append(float(few[9]))
        dicts['Smse'].append(float(few[4]))
        dicts['SG_mean'].append(float(few[11][:-1]))

        dicts['Mmae'].append(float(media[9]))
        dicts['Mmse'].append(float(media[4]))
        dicts['MG_mean'].append(float(media[11][:-1]))

        dicts['Lmae'].append(float(many[9]))
        dicts['Lmse'].append(float(many[4]))
        dicts['LG_mean'].append(float(many[11][:-1]))
        dicts['models'].append(i)
    except:
        dicts['models'].append(i)
        dicts['mae'].append("error")
        dicts['mse'].append("error")
        dicts['G_mean'].append("error")
        dicts['r2'].append("error")
        continue
df = pd.DataFrame(dicts)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--o', default="./ablition_res.xlsx", help='save dictionary')
    parser.set_defaults(augment=True)
    args, unknown = parser.parse_known_args()
    df.to_excel(args.o)