import Levenshtein
import subprocess
import re
import pandas as pd
from tqdm import tqdm

# Нахождение близких по расстоянию Левенштейна сиквенсов
def get_levenshtein_distance_sequences(data, active_data):
    min_distances = []
    min_sequences = []
    dot_bracket = []

    for sequence in tqdm(data['sequence'], desc="Levenshtein sequences progress"):
        lev_dist = []

        for active_sequence in active_data['sequence']:
            dist = Levenshtein.distance(sequence, active_sequence)
            max_len = max(len(sequence), len(active_sequence))
            norm_dist = dist / max_len
            lev_dist.append(norm_dist)

        min_idx = lev_dist.index(min(lev_dist))
        min_distances.append(lev_dist[min_idx])
        min_sequences.append(active_data.iloc[min_idx]['sequence']) # Самая близкая последовательность
        dot_bracket.append(active_data.iloc[min_idx]['db_structure']) # Ее dot-bracket структура

    return min_distances, min_sequences, dot_bracket

# Нахождение близких по расстоянию Левенштейна dot-bracket структур
def get_levenshtein_distance_dot_bracket(data, active_data):
    min_distances = []
    min_sequences = []
    dot_bracket = []

    for structure in tqdm(data['db_structure'], desc="Levenshtein dot-bracket progress"):
        lev_dist = []

        for active_structure in active_data['db_structure']:
            dist = Levenshtein.distance(structure, active_structure)
            max_len = max(len(structure), len(active_structure))
            norm_dist = dist / max_len
            lev_dist.append(norm_dist)

        min_idx = lev_dist.index(min(lev_dist))
        min_distances.append(lev_dist[min_idx])
        min_sequences.append(active_data.iloc[min_idx]['sequence']) # Ее нуклеотидная последовательность
        dot_bracket.append(active_data.iloc[min_idx]['db_structure']) # Самая близкая dot-bracket

    return min_distances, min_sequences, dot_bracket

def get_tree_edit_distance(data, active_data):
    min_distances = []
    min_sequences = []
    dot_bracket = []

    for structure in tqdm(data['db_structure'], desc="Tree Edit Distance progress"):
        ted_list = []

        for active_structure in active_data['db_structure']:
            input_data = f"{structure}\n{active_structure}\n"

            process = subprocess.Popen(
                ["RNAdistance", "-D", "f"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stdout, stderr = process.communicate(input=input_data)
            ted_list.append(int(stdout.split(":")[1].strip()))

        min_idx = ted_list.index(min(ted_list))
        min_distances.append(ted_list[min_idx])
        min_sequences.append(active_data.iloc[min_idx]['sequence'])
        dot_bracket.append(active_data.iloc[min_idx]['db_structure'])

    return min_distances, min_sequences, dot_bracket


def main(data_path, active_data_path, output_path):

    data = pd.read_csv(data_path)
    data = data[['sequence', 'db_structure']]
    active_data = pd.read_csv(active_data_path)

    min_lev_by_seq, min_lev_by_seq_seq,  min_lev_by_seq_dbs = get_levenshtein_distance_sequences(data, active_data)
    min_lev_by_dbs, min_lev_by_dbs_seq, min_lev_by_dbs_dbs = get_levenshtein_distance_dot_bracket(data, active_data)
    min_tde_dist, min_tde_seq, min_tde_db = get_tree_edit_distance(data, active_data)
    
    data['min_levenstein_by_seq'] = min_lev_by_seq
    data['min_levenshtein_by_seq_seq'] = min_lev_by_seq_seq
    data['min_levenshtein_by_seq_dbs'] = min_lev_by_seq_dbs
    data['min_levenstein_by_dbs'] = min_lev_by_dbs
    data['min_levenshtein_by_dbs_seq'] = min_lev_by_dbs_seq
    data['min_levenshtein_by_dbs_dbs'] = min_lev_by_dbs_dbs
    data['min_tde'] = min_tde_dist
    data['min_tde_sequence'] = min_tde_seq
    data['min_tde_db'] = min_tde_db

    data.to_csv(output_path, index=False)

    print('Finished')

if __name__ == '__main__':

    main(
        data_path = r'after_val_data/wgan_real_after_val.csv', # Сгенерированные сиквенсы
        active_data_path = r'data/active_seq_stats.csv', # Активные ДНКзимы
        output_path = 'similar_sequences_after_val/wgan_real.csv'
    )

    