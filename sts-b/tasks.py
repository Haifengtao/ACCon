import os
import nltk
import codecs
import logging
import pickle
import numpy as np
from scipy.ndimage import convolve1d
from util import get_lds_kernel_window, STSShotAverage

def process_sentence(sent, max_seq_len):
    '''process a sentence using NLTK toolkit'''
    return nltk.word_tokenize(sent)[:max_seq_len]

def load_tsv(data_file, max_seq_len, s1_idx=0, s2_idx=1, targ_idx=2, targ_fn=None, skip_rows=0, delimiter='\t', args=None):
    '''Load a tsv '''
    sent1s, sent2s, targs = [], [], []
    with codecs.open(data_file, 'r', 'utf-8') as data_fh:
        for _ in range(skip_rows):
            data_fh.readline()
        for row_idx, row in enumerate(data_fh):
            try:
                row = row.strip().split(delimiter)
                sent1 = process_sentence(row[s1_idx], max_seq_len)
                if (targ_idx is not None and not row[targ_idx]) or not len(sent1):
                    continue

                if targ_idx is not None:
                    targ = targ_fn(row[targ_idx])
                else:
                    targ = 0

                if s2_idx is not None:
                    sent2 = process_sentence(row[s2_idx], max_seq_len)
                    if not len(sent2):
                        continue
                    sent2s.append(sent2)

                sent1s.append(sent1)
                targs.append(targ)

            except Exception as e:
                logging.info(e, " file: %s, row: %d" % (data_file, row_idx))
                continue

    if args is not None:
        return_arr = (sent1s, sent2s, targs)

        # assert args.reweight in {'inverse', 'sqrt_inv'}
        # assert args.reweight != 'none' if args.lds else True, "Set reweight to \'inverse\' (default) or \'sqrt_inv\' when using LDS"

        bins = args.bucket_num
        value_lst, bins_edges = np.histogram(targs, bins=bins, range=(0., 5.))

        def get_bin_idx(label):
            if label == 5.:
                return bins - 1
            else:
                return np.where(bins_edges > label)[0][0] - 1

        if args.reweight == 'sqrt_inv':
            value_lst = [np.sqrt(x) for x in value_lst]
        num_per_label = [value_lst[get_bin_idx(label)] for label in targs]

        logging.info(f"Using re-weighting: [{args.reweight.upper()}]")

        if args.lds:
            lds_kernel_window = get_lds_kernel_window(args.lds_kernel, args.lds_ks, args.lds_sigma)
            logging.info(f'Using LDS: [{args.lds_kernel.upper()}] ({args.lds_ks}/{args.lds_sigma})')
            smoothed_value = convolve1d(value_lst, weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[get_bin_idx(label)] for label in targs]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return_arr+=(weights,)

        # import pdb
        # pdb.set_trace()
        print(args.regularization_type)
        import pandas as pd
        data = pd.DataFrame(
            {"age": targs}
        )
        data["age_bkt"] = data["age"] // 0.1  # // 0.02 // 0.01
        EF_freq = data['age_bkt'].value_counts(dropna=False).rename_axis('age_bkt_key').reset_index(name='counts')
        EF_freq = EF_freq.sort_values(by=['age_bkt_key']).reset_index()
        EF_dict = {}
        for key_itr_idx in range(len(EF_freq['age_bkt_key'])):
            if key_itr_idx == 0:
                EF_dict[EF_freq['age_bkt_key'][key_itr_idx]] = EF_freq['counts'][key_itr_idx]
            else:
                EF_dict[EF_freq['age_bkt_key'][key_itr_idx]] = EF_dict[EF_freq['age_bkt_key'][key_itr_idx - 1]] + \
                                                               EF_freq['age_bkt_key'][key_itr_idx]
        for key_itr_idx in range(len(EF_freq['age_bkt_key'])):
            EF_dict[EF_freq['age_bkt_key'][key_itr_idx]] = EF_dict[EF_freq['age_bkt_key'][key_itr_idx]] - \
                                                           EF_freq['age_bkt_key'][key_itr_idx] / 2
        data['age_CLS'] = data["age_bkt"].apply(lambda x: EF_dict[x])
        age_pdf = data['age_CLS'].to_list()

        if args.regularization_type == "ada" and args.reweight != 'none':
            return sent1s, sent2s, targs, weights, age_pdf
        elif args.regularization_type == "ada":
            return sent1s, sent2s, targs, age_pdf
        elif args.reweight != 'none':
            return sent1s, sent2s, targs, weights
        else:
            return sent1s, sent2s, targs
    return sent1s, sent2s, targs

class STSBTask:
    ''' Task class for Sentence Textual Similarity Benchmark.  '''
    def __init__(self, args, path, max_seq_len, name="sts-b"):
        ''' '''
        super(STSBTask, self).__init__()
        self.args = args
        self.name = name
        self.train_data_text, self.val_data_text, self.test_data_text = None, None, None
        self.val_metric = 'mse'
        self.scorer = STSShotAverage(metric=['mse', 'l1', 'gmean', 'pearsonr', 'spearmanr'])
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' '''
        print(self.args)
        if self.args.datatype == "balanced":
            tr_data = load_tsv(os.path.join(path, 'train_new.tsv'), max_seq_len, skip_rows=1,
                               s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: np.float32(x), args=self.args)
            val_data = load_tsv(os.path.join(path, 'dev_new.tsv'), max_seq_len, skip_rows=1,
                                s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: np.float32(x))
            te_data = load_tsv(os.path.join(path, 'test_new.tsv'), max_seq_len, skip_rows=1,
                               s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: np.float32(x))
        else:
            with open(os.path.join(path, 'train_natural.pkl'), 'rb') as f:
                tr_data = pickle.load(f)
            with open(os.path.join(path, 'dev_natural.pkl'), 'rb') as f:
                val_data = pickle.load(f)
            with open(os.path.join(path, 'test_natural.pkl'), 'rb') as f:
                te_data = pickle.load(f)
            if self.args is not None:
                sent1s, sent2s, targs = tr_data

                # assert args.reweight in {'inverse', 'sqrt_inv'}
                # assert args.reweight != 'none' if args.lds else True, "Set reweight to \'inverse\' (default) or \'sqrt_inv\' when using LDS"

                bins = self.args.bucket_num
                value_lst, bins_edges = np.histogram(targs, bins=bins, range=(0., 5.))

                def get_bin_idx(label):
                    if label == 5.:
                        return bins - 1
                    else:
                        return np.where(bins_edges > label)[0][0] - 1

                if self.args.reweight == 'sqrt_inv':
                    value_lst = [np.sqrt(x) for x in value_lst]
                num_per_label = [value_lst[get_bin_idx(label)] for label in targs]

                logging.info(f"Using re-weighting: [{self.args.reweight.upper()}]")

                if self.args.lds:
                    lds_kernel_window = get_lds_kernel_window(self.args.lds_kernel, self.args.lds_ks, self.args.lds_sigma)
                    logging.info(f'Using LDS: [{self.args.lds_kernel.upper()}] ({self.args.lds_ks}/{self.args.lds_sigma})')
                    smoothed_value = convolve1d(value_lst, weights=lds_kernel_window, mode='constant')
                    num_per_label = [smoothed_value[get_bin_idx(label)] for label in targs]

                weights = [np.float32(1 / x) for x in num_per_label]
                scaling = len(weights) / np.sum(weights)
                weights = [scaling * x for x in weights]
                # return_arr += (weights,)

                # import pdb
                # pdb.set_trace()
                print(self.args.regularization_type)
                import pandas as pd
                data = pd.DataFrame(
                    {"age": targs}
                )
                data["age_bkt"] = data["age"] // 0.1  # // 0.02 // 0.01
                EF_freq = data['age_bkt'].value_counts(dropna=False).rename_axis('age_bkt_key').reset_index(
                    name='counts')
                EF_freq = EF_freq.sort_values(by=['age_bkt_key']).reset_index()
                EF_dict = {}
                for key_itr_idx in range(len(EF_freq['age_bkt_key'])):
                    if key_itr_idx == 0:
                        EF_dict[EF_freq['age_bkt_key'][key_itr_idx]] = EF_freq['counts'][key_itr_idx]
                    else:
                        EF_dict[EF_freq['age_bkt_key'][key_itr_idx]] = EF_dict[
                                                                           EF_freq['age_bkt_key'][key_itr_idx - 1]] + \
                                                                       EF_freq['age_bkt_key'][key_itr_idx]
                for key_itr_idx in range(len(EF_freq['age_bkt_key'])):
                    EF_dict[EF_freq['age_bkt_key'][key_itr_idx]] = EF_dict[EF_freq['age_bkt_key'][key_itr_idx]] - \
                                                                   EF_freq['age_bkt_key'][key_itr_idx] / 2
                data['age_CLS'] = data["age_bkt"].apply(lambda x: EF_dict[x])
                age_pdf = data['age_CLS'].to_list()
            if self.args.regularization_type == "ada" and self.args.reweight != 'none':
                tr_data = (sent1s, sent2s, targs, weights, age_pdf)
            elif self.args.regularization_type == "ada":
                tr_data = ( sent1s, sent2s, targs, age_pdf)
            elif self.args.reweight != 'none':
                tr_data = ( sent1s, sent2s, targs, weights)
            else:
                tr_data = ( sent1s, sent2s, targs)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        logging.info("\tFinished loading STS Benchmark data.")

    def get_metrics(self, reset=False, type=None):
        metric = self.scorer.get_metric(reset, type)
        
        return metric


class STSBTask_Natural:
    ''' Task class for Sentence Textual Similarity Benchmark.  '''

    def __init__(self, args, path, max_seq_len, name="sts-b-natural"):
        ''' '''
        super(STSBTask_Natural, self).__init__()
        self.args = args
        self.name = name
        self.train_data_text, self.val_data_text, self.test_data_text = None, None, None
        self.val_metric = 'mse'
        self.scorer = STSShotAverage(metric=['mse', 'l1', 'gmean', 'pearsonr', 'spearmanr'])
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' '''
        print(self.args)
        with open(os.path.join(path, 'train_natural.pkl'), 'rb') as f:
            tr_data = pickle.load(f)
        with open(os.path.join(path, 'dev_natural.pkl'), 'rb') as f:
            val_data = pickle.load(f)
        with open(os.path.join(path, 'test_natural.pkl'), 'rb') as f:
            te_data = pickle.load(f)
        # tr_data = load_tsv(os.path.join(path, 'train_new.tsv'), max_seq_len, skip_rows=1,
        #                    s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: np.float32(x), args=self.args)
        # val_data = load_tsv(os.path.join(path, 'dev_new.tsv'), max_seq_len, skip_rows=1,
        #                     s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: np.float32(x))
        # te_data = load_tsv(os.path.join(path, 'test_new.tsv'), max_seq_len, skip_rows=1,
        #                    s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: np.float32(x))

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        logging.info("\tFinished loading STS Natural Benchmark data.")

    def get_metrics(self, reset=False, type=None):
        metric = self.scorer.get_metric(reset, type)

        return metric
