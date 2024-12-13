import os
import pandas as pd


def collect_res(datadir, save_dir):
    dicts = {"models": [], "mae": [], "mse": [], "G_mean": [],}# "r2": []}
    files = os.listdir(datadir)
    for i in files:
        with open(os.path.join(datadir, i, "training.log")) as f:
            strs = f.readlines()

        """
        2023-12-21 20:49:22,954 |  * Overall: MSE 2147.434       RMSE 46.340    L1 42.984       r2 -6.160
        2023-12-21 20:49:22,954 | Test loss: MSE [2147.4341], L1 [42.9840], G-Mean [39.2502]
        Done
        """
        try:
            # print(i)
            test_str = strs[-2][26:].split(" ")
            print(i, test_str)
            mse = float(test_str[3][1:-2])
            mae = float(test_str[5][1:-2])
            G_mean = float(test_str[-1][1:-2])

            # test_str2 = strs[-3][26:].split(" ")
            # r2 = float(test_str2[-1][:-1])

            # print(mae, mse, G_mean, r2)
            # items = [i, mae, mse, G_mean, r2]
            dicts['models'].append(i)
            dicts['mae'].append(mae)
            dicts['mse'].append(mse)
            dicts['G_mean'].append(G_mean)
            # dicts['r2'].append(r2)
        except Exception as e:
            print(e)
            for k in dicts:
                if k != 'models':
                    dicts[k].append('nan')
            dicts['models'].append(i)
            continue
    df = pd.DataFrame(dicts)
    df.to_excel(save_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', default="checkpoint", help='save dictionary')
    parser.add_argument('--o', default="./ablition_natural.xlsx", help='save dictionary')
    parser.set_defaults(augment=True)
    args, unknown = parser.parse_known_args()
    datadir = args.dir
    save_dir = args.o
    collect_res(datadir, save_dir)