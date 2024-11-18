from pathlib import Path

SEEDS = [2, 64, 0, 10, 36]

PATH_PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_PROJECT_DATA = Path.joinpath(PATH_PROJECT_DIR, 'data')
PATH_PROJECT_DATA_RAW = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw')
PATH_PROJECT_DATA_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'preprocessed')
PATH_PROJECT_DATA_PREPROCESSED_SIGNAL = Path.joinpath(PATH_PROJECT_DIR, 'data', 'preprocessed', 'time_series')
PATH_PROJECT_DATA_PREPROCESSED_TEXT = Path.joinpath(PATH_PROJECT_DIR, 'data', 'preprocessed', 'text')
PATH_PROJECT_DATA_PREPROCESSED_TABULAR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'preprocessed', 'tabular')
PATH_PROJECT_DATA_PREPROCESSED_FUSION = Path.joinpath(PATH_PROJECT_DIR, 'data', 'preprocessed', 'fusion')
PATH_PROJECT_MODELS = Path.joinpath(PATH_PROJECT_DIR, 'models')
PATH_PROJECT_REPORTS = Path.joinpath(PATH_PROJECT_DIR, 'reports')
PATH_PROJECT_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'figures')
PATH_PROJECT_TABULAR_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports','tabular','metrics')
PATH_PROJECT_TABULAR_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports','tabular','figures')
PATH_PROJECT_REPORTS_SIGNAL = Path.joinpath(PATH_PROJECT_DIR, 'reports','time_series','metrics')
PATH_PROJECT_TEXT_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports','text')
PATH_PROJECT_FS_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports','FS')
PATH_PROJECT_FUSION_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports','fusion','metrics')
PATH_PROJECT_FUSION_METRICS_MODALITIES = Path.joinpath(PATH_PROJECT_DIR, 'reports','fusion','metrics','modalities')
PATH_PROJECT_FUSION_METRICS_GENDER_MALE = Path.joinpath(PATH_PROJECT_DIR, 'reports','fusion','metrics', 'gender',
                                                        'male')
PATH_PROJECT_FUSION_METRICS_GENDER_FEMALE = Path.joinpath(PATH_PROJECT_DIR, 'reports','fusion','metrics', 'gender',
                                                        'Female')
PATH_PROJECT_FUSION_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports','fusion','figures')
PATH_PROJECT_MATRIX = Path.joinpath(PATH_PROJECT_DIR, 'reports','confussion_matrix')

ptid_female =[199, 114,  95, 136,  22,  19,  50, 115,  77, 137,  64, 144,  4,
       139,  86,  65, 195, 184,  26,  84,  96, 110, 123,  43,  49, 157,
        88,  29, 159, 180,  13, 140, 149,  14, 138, 124, 129,  12, 141,
        75,  63,  45, 101,  37, 100,  60, 178, 155, 203,  20,  67, 171,
       189, 172, 188,  36,  33,  40, 168,   1, 182,  32,  87,   3, 156,
        78, 185,  31,  44, 151, 152, 146, 161, 143,  42,  59,  61, 106,
       117,  15,  10,  28, 166, 187, 197,  89, 193,  8,  2,  46, 133,
       126,  72, 147,  85]

ptid_male = [ 56, 201, 107,  47, 181,  53,  81,  90,   5,  62, 148, 145, 192,
       154,  79, 165,  21, 175, 173, 196,  57, 108, 164, 158, 174,  80,
       167,  97, 202,  41,  69, 132, 200, 179, 116,  91, 153,  24, 131,
       150,   7, 113,  27, 191, 177, 109,  52, 120,  58, 127,  17,  94,
       104,  35, 162,   6, 119,  16,  51,  55, 142, 163,  30,  25, 190,
        48, 125,  34, 102, 135,  11, 105,  76, 134,  93,  39,  98,   9,
       170,  18,  73,  82,  66, 103, 122, 169, 186,  74,  68, 118,  38,
        92,  54, 112, 130, 176, 121,  83,  99, 128,  71, 198,  23,  70,
       160, 111]

BBDD_HYPO = 'hypo'
BBDD_ID_HYPO_LIFESTYLE = 'BDemoLifeDiabHxMgmt'
BBDD_HYPO_LIFESTYLE = 'lifestyle'
BBDD_ID_HYPO_DEPRESSION = 'BGeriDepressScale'
BBDD_HYPO_DEPRESSION = 'depression'
BBDD_ID_HYPO_ATTITUDE = 'BBGAttitudeScale'
BBDD_HYPO_ATTITUDE = 'attitude'
BBDD_ID_HYPO_FEAR = 'BHypoFearSurvey'
BBDD_HYPO_FEAR = 'fear'
BBDD_HYPO_CGM = 'BDataCGM'
BBDD_ID_HYPO_UNAWARE = 'BHypoUnawareSurvey'
BBDD_HYPO_UNAWARE = 'unaware'
BBDD_HYPO_LABEL = 'BPtRoster'


dict_names={'Unaware': ['LowBGSympCat_ALW', 'Bel70PastMonNoSymp_N', 'ExtentSympLowBG_A', 'LowBGSympCat_STM', 'LowBGLostSymp',
             'FeelSympLowBG_60-69',  'ExtentSympLowBG_S', 'FeelSympLowBG_0-40', 'Bel70PastMonNoSymp_2-3TW'],

'Fear': ['AvoidAloneLowBG', 'WorryNoHelp', 'WorryNotRecLowBG', 'WorryReacAlone', 'EatFirstSignLowBG',
             'WorryPassOut',  'WorryReactDrive'],

'Totscores': ['SymbDigWTotCorr', 'FrailtySecWalkTotTimeSec', 'ReadCardCorrLens', 'GrPegNonTotTime', 'TrailMakBTotTime',
               'GrPegDomTotTime', 'TrailMakATotTime', 'FrailtyFirstWalkTotTimeSec','SymbDigOTotCorr',
               'BGVisit2', 'DukeSocSatScore'],

'CGM': [ 'cde', 'dee', 'edd', 'hgg', 'fgf', 'bbc', 'dcc', 'bcd'],

'Medications' : ['medications3_2', 'medications3_0'],
}