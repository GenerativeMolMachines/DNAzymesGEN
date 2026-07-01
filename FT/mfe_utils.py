"""MFE для DNA через NUPACK с базовыми параметрами (material=\"dna\")."""

import re

from nupack import Complex, ComplexSet, Model, SetSpec, Strand, complex_analysis

DNA_MODEL = Model(material="dna")


def calculate_mfe(dna_sequence: str) -> float:
    """Минимальная свободная энергия одного штамма DNA (kcal/mol, NUPACK)."""
    valid_part = dna_sequence.split("N")[0].strip().upper()
    if not valid_part or not re.fullmatch(r"[ACGT]+", valid_part):
        return 0.0

    try:
        strand = Strand(valid_part, name="a")
        my_complex = Complex([strand])
        my_set = ComplexSet(strands={strand: 1e-6}, complexes=SetSpec(max_size=1))
        result = complex_analysis(my_set, compute=["mfe"], model=DNA_MODEL)
        return float(result[my_complex].mfe[0].energy)
    except Exception as e:
        print(f"MFE calculation error: {str(e)}")
        return 0.0
