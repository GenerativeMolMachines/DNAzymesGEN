import pandas as pd 
import Levenshtein
import warnings 
warnings.filterwarnings('ignore')


PATH_TO_ACTIVE_SEQ = 'data/active_DNAzymes.csv'
PATH_TO_GENERATED_SEQ = 'wgan_negative_based_with_stats_classification_results.csv'
PATH_TO_OUTPUT = 'wgan_negative_based_with_stats_25q.csv'


def read_data(active_data, generated_data):
    # Active sequences from SequenceCraft
    active_df = pd.read_csv(active_data)
    # Generated sequences + filter by probability 
    gen_df = pd.read_csv(generated_data)
    gen_df['y_pred'] = (gen_df['y_proba'] >= 0.95).astype(int)
    gen_df = gen_df[gen_df['y_pred'] == 1]
    gen_df.reset_index(inplace=True, drop=True)

    return active_df, gen_df 


def get_levenshtein_info(active_data, generated_data):
    lev_total_list = []
    for sequence in generated_data['sequence']:
        lev_dist = []
        for active_sequence in active_data['sequence']:
            dist = Levenshtein.distance(sequence, active_sequence)
            max_len = max(len(sequence), len(active_sequence))
            norm_dist = dist / max_len

            lev_dist.append(norm_dist)
        lev_total_list.append(lev_dist)

    return lev_total_list


def main():

    active_df, gen_df = read_data(
        active_data=PATH_TO_ACTIVE_SEQ,
        generated_data=PATH_TO_GENERATED_SEQ)
    
    gen_df['Levenshtein'] = get_levenshtein_info(
        active_data = active_df,
        generated_data=gen_df)
    
    gen_df['Levenshtein_min'] = [min(lst) for lst in gen_df['Levenshtein']]

    # Selection of 25th quantile
    q25 = gen_df['Levenshtein_min'].quantile(0.25)
    q75 = gen_df['Levenshtein_min'].quantile(0.75)
    df_25 = gen_df[gen_df['Levenshtein_min'] <= q25]
    df_iqr = gen_df[(gen_df['Levenshtein_min'] > q25) & (gen_df['Levenshtein_min'] <= q75)]

    df_25.drop(['y_pred', 'y_proba'], axis = 1, inplace = True)
    df_iqr.drop(['y_pred', 'y_proba'], axis = 1, inplace = True)

    df_25.to_csv(PATH_TO_OUTPUT)
    print(f'Data saved to {PATH_TO_OUTPUT}')


if __name__ == '__main__':
    main()

