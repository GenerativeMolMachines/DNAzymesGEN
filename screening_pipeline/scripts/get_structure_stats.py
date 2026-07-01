import pandas as pd
import forgi.graph.bulge_graph as fgb


# основная функция расчета статистики: петли, стебли и тд
def analyze_with_forgi(sequence: str, structure: str):
    
    bg = fgb.BulgeGraph.from_dotbracket(structure, seq=sequence)
    
    stats = {
        'seq': [],
        'db_structure': [],
        'stems': [],
        'hairpin_loops': [],
        'internal_loops': [],
        'multiloops': [],
        'fiveprime': [],
        'threeprime': []
    }
    
    stats['seq'].append(sequence)
    stats['db_structure'].append(structure)

    for element in bg.defines:
        element_type = element[0]  
        positions = bg.defines[element]
        
        if element_type == 's':  
            stats['stems'].append({
                'name': element,
                'positions': positions,
                'length': bg.stem_length(element),
                'sequence': bg.get_define_seq_str(element)
            })
        elif element_type == 'h':  
            stats['hairpin_loops'].append({
                'name': element,
                'positions': positions,
                'sequence': bg.get_define_seq_str(element)
            })
        elif element_type == 'i':  
            stats['internal_loops'].append({
                'name': element,
                'positions': positions,
                'sequence': bg.get_define_seq_str(element)
            })
        elif element_type == 'm': 
            stats['multiloops'].append({
                'name': element,
                'positions': positions
            })
        elif element_type == 'f':  
            stats['fiveprime'].append({
                'positions': positions,
                'sequence': bg.get_define_seq_str(element)
            })
        elif element_type == 't':  
            stats['threeprime'].append({
                'positions': positions,
                'sequence': bg.get_define_seq_str(element)
            })
    
    return stats


def create_summary_dataframe(stats_list):

    data_rows = []

    for stats in stats_list:
        
        sequence = ''.join(stats['seq'])
        dot_bracket = ''.join(stats['db_structure'])
        
        stems = stats.get('stems', [])
        stem_lens = [item['length'] for item in stems]
        
        stem_seqs_1 = [item['sequence'][0] for item in stems] 
        stem_seqs_2 = [item['sequence'][1] for item in stems] 

        hairpins = stats.get('hairpin_loops', [])

        hairpin_seqs = [item['sequence'][0] for item in hairpins]
        harpins_lens = [len(seq) for seq in hairpin_seqs]
        harpin_position = [item['positions'] for item in hairpins]

        row = {
            'sequence': sequence,
            'db_structure': dot_bracket,
            'stems_count': len(stems),
            'stems_lengths': stem_lens,
            'stems_seq_first': stem_seqs_1,
            'stems_seq_second': stem_seqs_2,
            'hairpins_count': len(hairpins),
            'harpins_lengths': harpins_lens, 
            'hairpins_sequences': hairpin_seqs,
            'harpin_position': harpin_position
        }
        
        data_rows.append(row)

    return pd.DataFrame(data_rows)



def count_mismatches(first_list, second_list):

    complement = {
        'A': ['T', 'U'], 
        'T': ['A'], 
        'U': ['A'], 
        'G': ['C'], 
        'C': ['G']
    }
    
    if isinstance(first_list, str):
        import ast
        first_list = ast.literal_eval(first_list)
    if isinstance(second_list, str):
        import ast
        second_list = ast.literal_eval(second_list)
    
    total_mismatches = 0
    for first_seq, second_seq in zip(first_list, second_list):
        second_seq_rev = second_seq[::-1]
        for n1, n2 in zip(first_seq, second_seq_rev):
            if n2 not in complement.get(n1, []):
                total_mismatches += 1
    
    return total_mismatches


def main(
    input_path: str,
    output_path: str      
):

    data = pd.read_csv(input_path)

    stats = []
    for i in range(len(data)):
        info = analyze_with_forgi(data['sequence'].loc[i], data['dot_bracket_structure'].loc[i])
        stats.append(info)

    results = create_summary_dataframe(stats)
    results['mismatch'] = results.apply(
    lambda row: count_mismatches(row['stems_seq_first'], row['stems_seq_second']), axis=1)

    results.to_csv(output_path)


if __name__ == '__main__':
    main(
        input_path="path",
        output_path="path"
    )
