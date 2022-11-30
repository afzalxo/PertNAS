from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [ #None op accomodated manually inside model
    'skip_connect',
    'max_pool_3x3',
    'avg_pool_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

PDARTS = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0),('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3',0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

ADA = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0),('skip_connect', 2), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5',3), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 1), ('skip_connect', 2), ('dil_conv_5x5', 1), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

NEW_ADA_2 = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0),('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3',0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

EPS_GREEDY = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0),('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3',2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6))

PASS_S2_WITHOUT_FIXED_VAL = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 3), ('skip_connect',3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))

DARTS_NOISE_EXTRACTED_FROM_G_2PASSES = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0),('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect',3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 3), ('sep_conv_5x5', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))

DARTS_NOISE_EXTRACTED_FROM_G_2PASSES_NEW_TOP = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0),('sep_conv_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3',1), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))

S2_NO_TOPOLOGY_3NODES = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_3x3',1), ('sep_conv_3x3', 2), ('skip_connect', 3)], normal_concat=range(2, 5), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 5))


DARTS_NOISE_NO_TOPOLOGY_3NODES = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0),('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3',1), ('avg_pool_3x3', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 5), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 3)], reduce_concat=range(2, 5))

#Err rate = 2.98
SIMPLIFIED_DARTS_RETHINKING_G_NEW_PERTURB = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),('sep_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('sep_conv_3x3',1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3)], normal_concat=range(2, 5), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('skip_connect', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 3)], reduce_concat=range(2, 5))

SIMPLIFIED_DARTS_RETHINKING_NEW_PERTURB = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3',1), ('max_pool_3x3', 2), ('dil_conv_3x3', 3)], normal_concat=range(2, 5), reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 5))

SIMPLIFIED_DARTS_RETHINKING_G_MULTIPLEPASSES_NEW_PERTURB = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3',1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3)], normal_concat=range(2, 5), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 5))

#Err rate = 2.65%
G_PHASED_OPCAP2_FT50_A9_SIMPLIFIED_DARTS = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect',3), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

#Err rate = 2.59%
G_PHASED_OPCAP1_FT50_A9_S2 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2),('sep_conv_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0),('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))

#Err rate = 2.84%
OPCAP4_FT10_VP1_DARTS = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

#Err rate = 3.07%
OPCAP4_FT25_VP05_DARTS = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 2), ('dil_conv_5x5', 3), ('skip_connect', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6)) 

OPCAP4_FT102535_VP1_DARTS = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 4)], reduce_concat=range(2, 6))

OPCAP4_FT101035_VP199_DARTS = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('skip_connect', 4)], reduce_concat=range(2, 6))

OPCAP4_SEQEDG_FT102535_VP1_DARTS = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

OPCAP4_SEQEDG_FT102535_VP199_SCRATCH_DARTS = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

TOPSEL_MAXIMP_OPCAP4_SEQEDG_FT102535_VP199_SCRATCH_DARTS = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6)) 

OPCAP4_SEQEDG_FT102535_VP199_SCRATCH_DARTS_RUN2 = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

OPCAP4_FT102535_VP199_SCRATCH_DARTS_RUN2 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6)) 

OPCAP2_FT102535_VP199_SCRATCH_DARTS = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('dil_conv_5x5', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6)) 

OPCAP1_FT102535_VP199_SCRATCH_C100_S3 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

OPCAP1_FT102535_VP199_SCRATCH_C100_S2 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6))

OPCAP1_FT102535_VP199_SCRATCH_C100_S4 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

OPCAP1_FT102535_VP199_SCRATCH_C10_S4 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

OPCAP2_FT102535_VP199_SCRATCH_DARTS_RERUN1 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

OPCAP2_FT102535_VP199_SCRATCH_DARTS_RERUN2 = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 3), ('avg_pool_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

OPCAP2_FT102535_VP199_SCRATCH_DARTS_RERUN5 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

OPCAP7_FT102535_VP199_SCRATCH_DARTS_PERT5 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 2), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))

OPCAP7_FT102535_VP199_SCRATCH_DARTS_PERT10 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

OPCAP2_VP05_PERT35_DARTS = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

OPCAP7_VP199_DARTS_PERT3 = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('skip_connect', 2), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))

OPCAP1_VP199_S1_PERT5_C10 = Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('max_pool_3x3', 1), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('skip_connect', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

OPCAP1_VP199_S1_PERT5_C100 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

OPCAP7_DARTS_FT102535_1PASS = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

OPCAP7_DARTS_FT102535_1PASS_RUN2 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

OPCAP4_DARTS_FT102535_2GPU_RUN3 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

OPCAP4_DARTS_FT102535_2GPU_RUN4 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

EVAL_BEST_EVOLUTIONARY = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

EVAL_BEST_SUPERNET = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

PCDARTS_CIFAR = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

