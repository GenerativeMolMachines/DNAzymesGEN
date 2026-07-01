import pandas as pd
import numpy as np
import warnings
import ast

MIN_HARPIN_LENGTH = 3
MAX_HARPIN_LENGTH = 12

MIN_STREM_LENGTH = 1
MAX_STREM_LENGTH = 9

LOWER_BOUND = 0.236
UPPER_BOUND = 0.791



def only_dots(data):

    only_dots_count = data['db_structure'].apply(lambda x: set(x) <= {'.'}).sum()
    data = data[~data['db_structure'].apply(lambda x: set(x) <= {'.'})]
    print(f'Количество последовательностей без шпилек (только точки): {only_dots_count}')
    return data


def harpins_outlier(data):

    before = data.shape[0]
    data = data[data['harpins_lengths'].apply(
        lambda lengths: all(MIN_HARPIN_LENGTH <= l <= MAX_HARPIN_LENGTH for l in lengths)
    )]
    after = data.shape[0]
    print(f'Количество сиквенсов с петлями вне диапазона min/max: {before - after}')
    return data


def strems_outliers(data):

    before = data.shape[0]
    data = data[data['stems_lengths'].apply(
        lambda lengths: all(MIN_STREM_LENGTH <= l <= MAX_STREM_LENGTH for l in lengths)
    )]
    after = data.shape[0]
    print(f'Количество сиквенсов со стяблями вне диапазона min/max: {before - after}')
    return data


def harpins_positions(data):
    before = data.shape[0]
    filtered_rows = []
    for idx, row in data.iterrows():
        seq_len = len(row['sequence'])
        harpin_pos = row['harpin_position']
        if isinstance(harpin_pos, str):
                harpin_pos = ast.literal_eval(harpin_pos)

        norm_centers = [((start + end) / 2) / seq_len for start, end in harpin_pos]
        if all(LOWER_BOUND <= center <= UPPER_BOUND for center in norm_centers):
            filtered_rows.append(row)

    data = pd.DataFrame(filtered_rows)
    after = data.shape[0]
    print(f'Последовательности в которых шпилька в аномальных положениях: {before - after}')
    return data


def main(
    input_path: str,
    output_path: str
):
    data = pd.read_csv(input_path)
    before_val = data.shape[0]

    data['stems_lengths'] = data['stems_lengths'].apply(ast.literal_eval)
    data['harpins_lengths'] = data['harpins_lengths'].apply(ast.literal_eval)

    dot_filtered = only_dots(data)
    harpins_filtered = harpins_outlier(dot_filtered)
    strems_filtered = strems_outliers(harpins_filtered)
    harpins_pos = harpins_positions(strems_filtered)
    after_val = harpins_pos.shape[0]
    print(f'Во время валидации удалено {before_val - after_val} последовательностей')

    harpins_pos.to_csv(output_path)


if __name__ == '__main__':
     main(
          input_path = "results/wgan_negative_based_stats.csv", 
          output_path = "test.csv"
     )