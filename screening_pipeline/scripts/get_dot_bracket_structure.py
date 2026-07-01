import os
import csv
from tqdm import tqdm
from typing import List, Tuple, Optional, Set
import numpy as np
from Bio.PDB import MMCIFParser
from pathlib import Path
current_dir = Path('af3_out')
    
base_folders = [p.name for p in current_dir.iterdir() if p.is_dir() and p.name.endswith("_apo")]
base_folders.sort()

print(f"Найдено папок: {len(base_folders)}")

from collections import defaultdict
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("apo_processing_log.txt", encoding='utf-8'),
        logging.StreamHandler()  
    ]
)
logger = logging.getLogger()

HBOND_CUTOFF, MIN_LOOP, ALLOW_WOBBLE = 3.6, 3, True
DNA_3TO1 = {"DA": "A", "DT": "T", "DG": "G", "DC": "C"}


def _base_letter(res) -> Optional[str]:

    name = (res.get_resname() or "").strip().upper()
    return DNA_3TO1.get(name)


def _coord(res,
           atom_name: str) -> Optional[np.ndarray]:

    try:
        return np.asarray(res[atom_name].coord, dtype=float)
    except KeyError:
        return None


def _dist(a: Optional[np.ndarray],
          b: Optional[np.ndarray]) -> Optional[float]:

    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def _contact(resA,
             resB,
             nameA: str,
             nameB: str,
             cutoff: float) -> Optional[float]:

    d = _dist(_coord(resA, nameA), _coord(resB, nameB))
    if d is None or d > cutoff:
        return None
    return cutoff - d


def _wc_score(res_i,
              base_i: str,
              res_j,
              base_j: str,
              cutoff: float,
              allow_wobble: bool) -> float:

    if {base_i, base_j} == {"A", "T"}:
        if base_i == "A":
            c1 = _contact(res_i, res_j, "N1", "N3", cutoff)
            c2 = _contact(res_i, res_j, "N6", "O4", cutoff)
        else:
            c1 = _contact(res_j, res_i, "N1", "N3", cutoff)
            c2 = _contact(res_j, res_i, "N6", "O4", cutoff)
        return (c1 or 0.0) + (c2 or 0.0) if (c1 is not None and c2 is not None) else 0.0

    if {base_i, base_j} == {"G", "C"}:

        if base_i == "G":
            contacts = [_contact(res_i, res_j, "N1", "N3", cutoff),
                        _contact(res_i, res_j, "O6", "N4", cutoff),
                        _contact(res_i, res_j, "N2", "O2", cutoff),]
        else:
            contacts = [_contact(res_j, res_i, "N1", "N3", cutoff),
                        _contact(res_j, res_i, "O6", "N4", cutoff),
                        _contact(res_j, res_i, "N2", "O2", cutoff),]

        valid = [c for c in contacts if c is not None]

        return sum(valid) if len(valid) >= 2 else 0.0


    # иногда бывают G-T пары
    if allow_wobble and ((base_i, base_j) in (("G","T"), ("T","G"))):
        if base_i == "G":
            c1 = _contact(res_i, res_j, "N1", "N3", cutoff)
            c2 = _contact(res_i, res_j, "O6", "O4", cutoff)
        else:
            c1 = _contact(res_j, res_i, "N1", "N3", cutoff)
            c2 = _contact(res_j, res_i, "O6", "O4", cutoff)
        if c1 is not None and c2 is not None:
            return 0.5 * ((c1 or 0.0) + (c2 or 0.0))
        return 0.0

    return 0.0


def calc_distance(coord1,
                  coord2):

    return np.linalg.norm(coord1 - coord2)


def is_base_pair(res1,
                 res2,
                 cutoff: float = HBOND_CUTOFF,
                 allow_wobble: bool = ALLOW_WOBBLE):

    b1 = _base_letter(res1)
    b2 = _base_letter(res2)
    if b1 is None or b2 is None:
        return False
    return _wc_score(res1, b1, res2, b2, cutoff, allow_wobble) > 0.0


def _nussinov_weighted(weights: np.ndarray,
                       min_loop: int) -> Set[Tuple[int, int]]:

    n = weights.shape[0]
    if n == 0:
        return set()
    S = np.zeros((n, n), dtype=float)

    for l in range(1, n):
        for i in range(0, n - l):
            j = i + l
            best = max(S[i+1, j], S[i, j-1])
            if j - i > min_loop and weights[i, j] > 0.0:
                best = max(best, S[i+1, j-1] + weights[i, j])

            for k in range(i+1, j):
                cand = S[i, k] + S[k+1, j]
                if cand > best:
                    best = cand
            S[i, j] = best

    pairs: Set[Tuple[int, int]] = set()
    def tb(i: int, j: int):
        if i >= j:
            return
        if np.isclose(S[i, j], S[i+1, j]):
            tb(i+1, j)
        elif np.isclose(S[i, j], S[i, j-1]):
            tb(i, j-1)
        elif j - i > min_loop and weights[i, j] > 0.0 and np.isclose(S[i, j], S[i+1, j-1] + weights[i, j]):
            pairs.add((i, j))
            tb(i+1, j-1)
        else:
            for k in range(i+1, j):
                if np.isclose(S[i, j], S[i, k] + S[k+1, j]):
                    tb(i, k)
                    tb(k+1, j)
                    break
    tb(0, n-1)
    return pairs


def generate_dot_bracket(residues,
                         base_pairs: Set[Tuple[int, int]]):

    n = len(residues)
    s = ['.'] * n
    for i, j in base_pairs:
        s[i] = '('
        s[j] = ')'
    return ''.join(s)

def _extract_dna_residues(chain) -> List:

    residues = []
    for res in chain:
        hetflag = res.id[0]
        if hetflag not in (' ', ''):
            continue
        if _base_letter(res) is None:
            continue
        residues.append(res)
    return residues

def _chain_dot_bracket(residues: List,
                       cutoff=HBOND_CUTOFF,
                       min_loop=MIN_LOOP,
                       allow_wobble=ALLOW_WOBBLE) -> str:
    n = len(residues)
    if n == 0:
        return ""
    bases = [_base_letter(r) for r in residues]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        if bases[i] is None:
            continue
        for j in range(i+1, n):
            if bases[j] is None:
                continue
            W[i, j] = _wc_score(residues[i], bases[i], residues[j], bases[j], cutoff, allow_wobble)
    pairs = _nussinov_weighted(W, min_loop=min_loop)
    return generate_dot_bracket(residues, pairs)

def process_cif_file(file_path: str) -> List[Tuple[str, str]]:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("DNA_structure", file_path)

    out: List[Tuple[str, str]] = []
    for model in structure:
        for chain in model:
            residues = _extract_dna_residues(chain)
            if not residues:
                continue

            sequence = ''.join(_base_letter(r) or 'N' for r in residues)
            dot = _chain_dot_bracket(residues, cutoff=HBOND_CUTOFF, min_loop=MIN_LOOP, allow_wobble=ALLOW_WOBBLE)

            if dot:
                out.append((dot, sequence))
    return out

def process_all_cif_files(base_folders: List[str], output_csv: str):
    results = []
    stats = defaultdict(int)  # для финальной сводки

    cwd = os.getcwd()

    for folder_name in tqdm(base_folders, desc='Обработка папок', unit='folder'):
        try:
            folder_index = int(folder_name.split('_')[0])  # если имя вида "123_seq_apo"
        except Exception:
            folder_index = -1

        folder_path = current_dir / folder_name
        if not folder_path.is_dir():
            logger.warning(f"[SKIP] Папка не найдена: {folder_name}")
            stats['folder_not_found'] += 1
            continue

        cif_files = list(folder_path.glob("*.cif")) + list(folder_path.glob("*.cif.gz"))
        if not cif_files:
            logger.info(f"[NO CIF] {folder_name} — нет .cif файлов")
            stats['no_cif_files'] += 1
            continue

        processed_any = False
        for cif_path in cif_files:
            try:
                dot_brackets = process_cif_file(str(cif_path))
            except Exception as e:
                logger.error(f"[ERROR] {folder_name}/{cif_path.name} — {e}")
                stats['parse_error'] += 1
                continue

            if not dot_brackets:
                logger.info(f"[NO DNA] {folder_name}/{cif_path.name} — ДНК не обнаружена или не распознана")
                stats['no_dna_found'] += 1

                continue

            for dot, seq in dot_brackets:
                if dot.strip('.') == '': 
                    logger.info(f"[UNFOLDED] {folder_name} | {seq[:30]}... → {dot}")
                    stats['unfolded_all_dots'] += 1
                else:
                    logger.info(f"[SUCCESS] {folder_name} | {seq[:30]}... → {dot}")
                    stats['success_with_structure'] += 1

                results.append([folder_index, seq, dot])
                processed_any = True

    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['index', 'sequence', 'dot_bracket_structure'])
        writer.writerows(results)

    logger.info("\n" + "="*60)
    logger.info("ИТОГОВАЯ СТАТИСТИКА ОБРАБОТКИ APO-СТРУКТУР")
    logger.info("="*60)
    logger.info(f"Всего папок _apo:              {len(base_folders)}")
    logger.info(f"Успешно обработано (есть структура): {stats['success_with_structure']}")
    logger.info(f"ДНК найдена, но полностью развёрнута: {stats['unfolded_all_dots']}")
    logger.info(f"ДНК вообще не обнаружена:       {stats['no_dna_found']}")
    logger.info(f"Нет .cif файлов:                {stats['no_cif_files']}")
    logger.info(f"Ошибки парсинга:                {stats['parse_error']}")
    logger.info(f"Папок без единой валидной структуры: {stats['folder_no_valid_structure']}")
    logger.info("="*60)

    print(f"\nГотово! Записано {len(results)} структур в {output_csv}")
    print(f"Полный лог сохранён в apo_processing_log.txt")

if __name__ == "__main__":

    output_csv = "discrete_fm_dbs.csv"
    process_all_cif_files(base_folders, output_csv)

    