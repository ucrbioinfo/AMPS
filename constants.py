from enum import Enum

NON_METH_TAG = 0.00000001
FLANKING_MARKER = 2


class AnnotTypes(Enum):
    gene = 'gene'
    exon = 'exon'
    CDS = 'CDS'
    mRNA = 'mRNA'

class ContextTypes():
    CG = 1
    CHG = 2
    CHH = 3

    cntx_str = {
        'CG': CG,
        'CHG': CHG,
        'CHH': CHH
    }

