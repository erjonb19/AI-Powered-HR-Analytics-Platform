# Sorvex 360 — Synthetic Data Pipeline V3
# Converted from Jupyter notebook

# # 🏗️ Sorvex 360 — Synthetic Candidate Pipeline V3
# **Predict • Prepare • Place • Post-Hire Outcomes**
# 
# ## Architecture
# - **Pure rule-based generation** — no CTGAN, no noise dilution
# - All 10,000 records generated directly from SOC-calibrated statistical distributions
# - Every correlation is intentional and preserved exactly as designed
# 
# ## Join Keys
# - `CandidateID` → links Phase 1, 2, 3
# - `PlacementID` → links Phase 3 to Phase 4
# 
# ## Outputs
# - `Sorvex360_Phase1_Predict.csv` — candidate assessment records
# - `Sorvex360_Phase2_Prepare.csv` — training program records
# - `Sorvex360_Phase3_Place.csv` — placement / hire records
# - `Sorvex360_Phase4_PostHire.csv` — outcome records (targets for ML)
# - `Sorvex360_Master_Flat.csv` — all phases joined flat (use this for ML training)
# 
# ---
# **Runtime:** ~60 seconds on any Colab instance (CPU is fine)

# ── Cell 1 — Install & Import ────────────────────────────────────────────────
!pip install pandas numpy scipy matplotlib --quiet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import random

warnings.filterwarnings('ignore')
np.random.seed(360)
random.seed(360)

OUTPUT_DIR = Path('/content/sorvex360_outputs/')
OUTPUT_DIR.mkdir(exist_ok=True)

N = 10_000  # Total records — pure rule-based, generated in ~60 seconds

print(f'✅ Ready. Generating {N:,} synthetic candidate records.')
print(f'Output directory: {OUTPUT_DIR}')

# ── Cell 2 — SOC Configuration ───────────────────────────────────────────────
# All score parameters, CDL rates, and role-specific settings per SOC
# Source: ONET skills/knowledge EDA + CareerBuilder analysis + Blueprint

SOC_CONFIG = {
    '49-2022.00': {
        'label':              'Telecom Equipment Installer/Repairer',
        'share':              0.28,
        'cognitive_mean':     68, 'cognitive_sd':     9,
        'simulation_mean':    72, 'simulation_sd':    9,
        'behavioral_mean':    67, 'behavioral_sd':    9,
        'situational_mean':   70, 'situational_sd':   8,
        'cdl_rate':           0.02,
        'union_rate':         0.08,
        'training_hrs_mean':  4500, 'training_hrs_sd': 1200,
        'physical_pass':      0.85,
        'loto_rate':          0.40,
        'cpr_rate':           0.06,
        'osha10_rate':        0.12,
        'pole_climb':         False,
        'base_safety_rate':   0.04,  # SOC-level OSHA incident base
        'base_tenure_yr':     3.9,
    },
    '49-9051.00': {
        'label':              'Power Line Installer/Repairer',
        'share':              0.20,
        'cognitive_mean':     67, 'cognitive_sd':     9,
        'simulation_mean':    71, 'simulation_sd':    9,
        'behavioral_mean':    68, 'behavioral_sd':    9,
        'situational_mean':   67, 'situational_sd':   9,
        'cdl_rate':           0.25,
        'union_rate':         0.22,
        'training_hrs_mean':  7000, 'training_hrs_sd': 1500,
        'physical_pass':      0.75,
        'loto_rate':          0.90,
        'cpr_rate':           0.08,
        'osha10_rate':        0.45,
        'pole_climb':         True,
        'base_safety_rate':   0.10,
        'base_tenure_yr':     3.9,
    },
    '51-8013.00': {
        'label':              'Power Plant Operator',
        'share':              0.16,
        'cognitive_mean':     75, 'cognitive_sd':     8,
        'simulation_mean':    76, 'simulation_sd':    8,
        'behavioral_mean':    72, 'behavioral_sd':    8,
        'situational_mean':   73, 'situational_sd':   8,
        'cdl_rate':           0.02,
        'union_rate':         0.18,
        'training_hrs_mean':  6000, 'training_hrs_sd': 1800,
        'physical_pass':      0.82,
        'loto_rate':          0.98,
        'cpr_rate':           0.18,
        'osha10_rate':        0.15,
        'pole_climb':         False,
        'base_safety_rate':   0.06,
        'base_tenure_yr':     4.9,
    },
    '51-8031.00': {
        'label':              'Water/Wastewater Treatment Plant Operator',
        'share':              0.16,
        'cognitive_mean':     72, 'cognitive_sd':     8,
        'simulation_mean':    74, 'simulation_sd':    8,
        'behavioral_mean':    71, 'behavioral_sd':    8,
        'situational_mean':   72, 'situational_sd':   8,
        'cdl_rate':           0.02,
        'union_rate':         0.12,
        'training_hrs_mean':  5500, 'training_hrs_sd': 1600,
        'physical_pass':      0.80,
        'loto_rate':          0.85,
        'cpr_rate':           0.35,
        'osha10_rate':        0.10,
        'pole_climb':         False,
        'base_safety_rate':   0.03,
        'base_tenure_yr':     4.9,
    },
    '51-8092.00': {
        'label':              'Gas Plant Operator',
        'share':              0.12,
        'cognitive_mean':     73, 'cognitive_sd':     8,
        'simulation_mean':    75, 'simulation_sd':    8,
        'behavioral_mean':    73, 'behavioral_sd':    8,
        'situational_mean':   72, 'situational_sd':   8,
        'cdl_rate':           0.40,
        'union_rate':         0.40,
        'training_hrs_mean':  6500, 'training_hrs_sd': 1700,
        'physical_pass':      0.82,
        'loto_rate':          0.95,
        'cpr_rate':           0.20,
        'osha10_rate':        0.15,
        'pole_climb':         False,
        'base_safety_rate':   0.08,
        'base_tenure_yr':     4.9,
    },
    '43-5041.00': {
        'label':              'Meter Reader, Utilities',
        'share':              0.08,
        'cognitive_mean':     63, 'cognitive_sd':     8,
        'simulation_mean':    58, 'simulation_sd':    9,
        'behavioral_mean':    70, 'behavioral_sd':    8,
        'situational_mean':   65, 'situational_sd':   8,
        'cdl_rate':           0.02,
        'union_rate':         0.10,
        'training_hrs_mean':  2500, 'training_hrs_sd': 800,
        'physical_pass':      0.92,
        'loto_rate':          0.20,
        'cpr_rate':           0.05,
        'osha10_rate':        0.08,
        'pole_climb':         False,
        'base_safety_rate':   0.01,
        'base_tenure_yr':     4.9,
    },
}

SOC_CODES  = list(SOC_CONFIG.keys())
SOC_SHARES = [SOC_CONFIG[s]['share'] for s in SOC_CODES]

print('✅ SOC config loaded:', len(SOC_CONFIG), 'roles')
for soc, cfg in SOC_CONFIG.items():
    print(f"  {soc} — {cfg['label']} ({cfg['share']*100:.0f}%)")

# ── Cell 3 — Helper Functions ────────────────────────────────────────────────

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def bernoulli(rate):
    return np.random.random() < rate

def normal_score(mean, sd, lo=50, hi=100):
    return int(clamp(round(np.random.normal(mean, sd)), lo, hi))

def random_date(start_year=2020, end_year=2024):
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    return start + timedelta(days=np.random.randint(0, (end - start).days))

def age_tenure_mean(age):
    """BLS median tenure by age band (years)"""
    if age < 25: return 1.0
    if age < 35: return 2.7
    if age < 45: return 4.6
    if age < 55: return 7.0
    return 9.6

def tenure_1yr_rate(age):
    """Probability of passing 1-year mark by age band (BLS)"""
    if age < 25: return 0.65
    if age < 35: return 0.72
    if age < 45: return 0.82
    if age < 55: return 0.87
    return 0.90

def compute_pi_score(cog, sim, beh, sit, has_prior, edu):
    """Sorvex360 PI Score — weighted composite (0-100)"""
    edu_bonus = {
        'HS/GED': 0,
        'Vocational Certificate': 3,
        'Associate Technical': 8,
        'Associate General': 4,
        "Bachelor's": 5,
        'Apprenticeship': 6,
    }
    score = (
        cog * 0.30 +
        beh * 0.20 +
        sit * 0.20 +
        sim * 0.15 +
        (10 if has_prior else 0) +
        edu_bonus.get(edu, 0)
    )
    return round(clamp(score, 0, 100), 1)

print('✅ Helper functions ready.')

# ── Cell 4 — Phase 1: Generate PREDICT Records ───────────────────────────────
print(f'Generating {N:,} Phase 1 (PREDICT) records...\n')

EDU_LEVELS   = ['HS/GED', 'Vocational Certificate', 'Associate Technical',
                'Associate General', "Bachelor's", 'Apprenticeship']
EDU_WEIGHTS  = [0.32, 0.18, 0.17, 0.06, 0.08, 0.19]

SOURCE_LABELS  = ['WorkforceProgram', 'SelfReferred', 'TradeUnion',
                  'MilitaryTransition', 'CommunityCollege', 'Other']
SOURCE_WEIGHTS = [0.25, 0.22, 0.20, 0.12, 0.13, 0.08]

INDUSTRY_LABELS  = ['Utility', 'Construction', 'Manufacturing', 'Military', 'Other']
INDUSTRY_WEIGHTS = [0.35, 0.25, 0.18, 0.12, 0.10]

TRAINING_LABELS  = ['IBEW JATC', 'Trade Union', 'Community College',
                    'Employer Program', 'Military', 'Self-taught']
TRAINING_WEIGHTS = [0.28, 0.20, 0.18, 0.15, 0.12, 0.07]

STATE_LABELS  = ['CA', 'TX', 'VA', 'NC', 'WA', 'FL', 'IL', 'PA', 'MI', 'MD', 'TN', 'Other']
STATE_WEIGHTS = [0.14, 0.13, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06, 0.05, 0.05, 0.04, 0.12]

phase1_records = []

for i in range(N):
    candidate_id = f'SORV-{i+1:05d}'
    soc = np.random.choice(SOC_CODES, p=SOC_SHARES)
    cfg = SOC_CONFIG[soc]

    # Demographics
    age    = int(clamp(round(np.random.normal(43.1, 10.5)), 18, 60))
    gender = np.random.choice(['Male', 'Female', 'Other'], p=[0.731, 0.259, 0.010])
    state  = np.random.choice(STATE_LABELS, p=STATE_WEIGHTS)
    lang   = np.random.choice(['English', 'Spanish', 'Other'], p=[0.813, 0.161, 0.026])
    vet    = bernoulli(0.12)

    # Education & Credentials
    edu    = np.random.choice(EDU_LEVELS, p=EDU_WEIGHTS)
    cdl    = bernoulli(cfg['cdl_rate'])
    osha10 = bernoulli(cfg['osha10_rate'])
    cpr    = bernoulli(cfg['cpr_rate'])

    # Assessment scores — SOC-specific distributions
    cog = normal_score(cfg['cognitive_mean'],   cfg['cognitive_sd'])
    sim = normal_score(cfg['simulation_mean'],  cfg['simulation_sd'])
    beh = normal_score(cfg['behavioral_mean'],  cfg['behavioral_sd'])
    sit = normal_score(cfg['situational_mean'], cfg['situational_sd'])

    # Score modifiers — empirically supported
    if age >= 50:
        cog = int(clamp(cog * 0.95, 50, 100))       # APA aging: ~5% decline
    if edu == 'Associate Technical':
        cog = int(clamp(cog * 1.05, 50, 100))       # CAST-R: +5% composite
    if vet:
        beh = int(clamp(beh * 1.04, 50, 100))       # Military discipline boost
        sit = int(clamp(sit * 1.03, 50, 100))

    # Prior work experience — age-banded rates (BLS)
    prior_exp_rate = {(18, 25): 0.30, (26, 35): 0.65, (36, 45): 0.82, (46, 61): 0.91}
    has_prior = False
    for (lo, hi), rate in prior_exp_rate.items():
        if lo <= age < hi:
            has_prior = bernoulli(rate)
            break

    # Job tenure history
    tenure_yrs = max(0.1, np.random.lognormal(
        np.log(max(0.1, age_tenure_mean(age))), 0.6
    ))
    longest_tenure = round(clamp(tenure_yrs, 0.1, 35), 1)

    # Screening eligibility
    has_license = bernoulli(0.78)
    drug_screen = bernoulli(0.88)
    background  = bernoulli(0.85)

    # PI Score
    pi_score = compute_pi_score(cog, sim, beh, sit, has_prior, edu)

    phase1_records.append({
        'CandidateID':             candidate_id,
        'SOC_Code':                soc,
        'JobFamily':               cfg['label'],
        'Age':                     age,
        'Gender':                  gender,
        'State':                   state,
        'PrimaryLanguage':         lang,
        'VeteranStatus':           int(vet),
        'EducationLevel':          edu,
        'CDL_Status':              int(cdl),
        'OSHA10_Status':           int(osha10),
        'CPR_Status':              int(cpr),
        'CognitiveScore':          cog,
        'SimulationScore':         sim,
        'BehavioralScore':         beh,
        'SituationalScore':        sit,
        'HasPriorTradeExperience': int(has_prior),
        'MostRecentIndustry':      np.random.choice(INDUSTRY_LABELS, p=INDUSTRY_WEIGHTS),
        'LongestJobTenure':        longest_tenure,
        'TrainingSource':          np.random.choice(TRAINING_LABELS, p=TRAINING_WEIGHTS),
        'ApprenticeshipInterest':  int(bernoulli(0.55)),
        'HasValidLicense':         int(has_license),
        'CanPassDrugScreen':       int(drug_screen),
        'CanPassBackgroundCheck':  int(background),
        'SourceOfCandidate':       np.random.choice(SOURCE_LABELS, p=SOURCE_WEIGHTS),
        'Sorvex360PI_Score':       pi_score,
    })

phase1_df = pd.DataFrame(phase1_records)
print(f'✅ Phase 1 complete: {len(phase1_df):,} records')
print(f'\nSOC distribution:')
print(phase1_df['SOC_Code'].value_counts())
print(f'\nPI Score stats:')
print(phase1_df['Sorvex360PI_Score'].describe().round(2))

# ── Cell 5 — Phase 2: Generate PREPARE Records ───────────────────────────────
print(f'Generating Phase 2 (PREPARE) records conditioned on Phase 1...\n')

EMPLOYER_NAMES = {
    '49-2022.00': ['Charter Communications', 'US Cellular', 'Intrado',
                   'Avacend Corporation', 'Leidos', 'General Dynamics IT'],
    '49-9051.00': ['Pike Corporation', 'NPL Construction Co.', 'Q3 Contracting Inc.',
                   'Davey Tree Expert Co.', 'Altec Industries', 'PDS Tech Inc.'],
    '51-8013.00': ['Exelon Corporation', 'BGIS North America', 'Exide Technologies',
                   'Veolia North America', 'General Dynamics IT'],
    '51-8031.00': ['Veolia North America', 'BGIS North America',
                   'NPL Construction Co.', 'Exelon Corporation'],
    '51-8092.00': ['Exelon Corporation', 'General Dynamics IT',
                   'Veolia North America', 'Leidos', 'GPAC'],
    '43-5041.00': ['Exelon Corporation', 'US Cellular',
                   'Charter Communications', 'Veolia North America'],
}

phase2_records = []

for _, p1 in phase1_df.iterrows():
    soc = p1['SOC_Code']
    cfg = SOC_CONFIG[soc]
    pi  = p1['Sorvex360PI_Score']
    age = p1['Age']

    # Training hours — SOC-specific, WA L&I distributions
    train_hrs = int(clamp(
        round(np.random.normal(cfg['training_hrs_mean'], cfg['training_hrs_sd'])),
        500, 10000
    ))

    # Attendance — Beta distribution conditioned on BehavioralScore
    beh_norm  = p1['BehavioralScore'] / 100
    att_mean  = 0.65 + beh_norm * 0.25   # maps behavioral 0.5-1.0 → attendance 0.65-0.90
    att_alpha = att_mean * 8
    att_beta  = (1 - att_mean) * 8
    attendance = round(clamp(np.random.beta(att_alpha, att_beta), 0.50, 0.99), 3)

    # Dates
    start_date = random_date(2020, 2024)
    end_date   = start_date + timedelta(weeks=train_hrs / 40)

    # Module pass — conditioned on attendance + cognitive
    if attendance > 0.85 and p1['CognitiveScore'] > 60:
        pass_prob = 0.78
    elif attendance > 0.75:
        pass_prob = 0.55
    elif p1['CognitiveScore'] > 65:
        pass_prob = 0.50
    else:
        pass_prob = 0.30
    passed_modules = bernoulli(pass_prob)

    # Certifications — conditioned on passed_modules
    certs = int(clamp(
        np.random.poisson(4.1 if passed_modules else 1.1),
        0, 6
    ))

    # Simulation performance — tracks SimulationScore with some noise
    sim_perf = clamp(round(np.random.normal(p1['SimulationScore'] * 0.9, 8)), 0, 100)

    # Instructor-rated scores
    teamwork = round(clamp(np.random.normal(3.2, 0.7), 1, 5), 1)

    # SafetyCommitmentScore — EEI SCL calibrated, +10% for experienced workers
    safety_base = round(clamp(
        np.random.normal(2.8 + p1['BehavioralScore'] / 100 * 1.5, 0.65), 1, 5
    ), 1)
    if p1['LongestJobTenure'] > 5:
        safety_base = round(clamp(safety_base * 1.10, 1, 5), 1)

    reliability = round(clamp(
        np.random.normal(3.3 + (attendance - 0.75) * 2, 0.65), 1, 5
    ), 1)

    # Physical tests
    phys_rate  = cfg['physical_pass'] * (0.92 if age >= 50 else 1.0)
    phys_pass  = bernoulli(phys_rate)
    pole_climb = bernoulli(0.72) if cfg['pole_climb'] else None
    lift50     = bernoulli(min(phys_rate * 1.05, 1.0))
    colorblind = bernoulli(0.08 if p1['Gender'] == 'Male' else 0.005)

    # Completion — WA L&I calibrated
    if passed_modules and attendance > 0.80:
        completed_prob = 0.85
    elif passed_modules:
        completed_prob = 0.65
    elif attendance > 0.75:
        completed_prob = 0.40
    else:
        completed_prob = 0.30
    completed = bernoulli(completed_prob)

    dropout_reason = None
    if not completed:
        dropout_reason = np.random.choice(
            ['Personal', 'Work Conflict', 'Performance', 'Medical', 'Unknown'],
            p=[0.35, 0.25, 0.20, 0.10, 0.10]
        )

    # PI Score at completion — training has real impact
    delta = np.random.normal(8, 5) if completed else np.random.normal(-8, 4)
    pi_at_completion = round(clamp(pi + delta, 0, 100), 1)

    phase2_records.append({
        'CandidateID':                    p1['CandidateID'],
        'TrainingProvider':               np.random.choice(EMPLOYER_NAMES[soc]),
        'StartDate':                      start_date.strftime('%Y-%m-%d'),
        'EndDate':                        end_date.strftime('%Y-%m-%d'),
        'TotalTrainingHours':             train_hrs,
        'AttendanceRate':                 attendance,
        'PassedRequiredModules':          int(passed_modules),
        'CertificationsEarned':           certs,
        'SimulationPerformance':          sim_perf,
        'TeamworkScore':                  teamwork,
        'SafetyCommitmentScore':          safety_base,
        'ReliabilityScore':               reliability,
        'PhysicalTestResult':             int(phys_pass),
        'PoleClimbResult':                int(pole_climb) if pole_climb is not None else None,
        'Lift50lbsTest':                  int(lift50),
        'ColorBlindness':                 int(colorblind),
        'Completed':                      int(completed),
        'DropoutReason':                  dropout_reason,
        'Sorvex360PI_Score_AtCompletion': pi_at_completion,
        'ReadinessDelta':                 round(pi_at_completion - pi, 1),
    })

phase2_df = pd.DataFrame(phase2_records)
print(f'✅ Phase 2 complete: {len(phase2_df):,} records')
print(f'Completion rate: {phase2_df["Completed"].mean():.1%}')
print(f'Avg attendance: {phase2_df["AttendanceRate"].mean():.3f}')
print(f'Avg ReadinessDelta: {phase2_df["ReadinessDelta"].mean():.2f}')

# ── Cell 6 — Phase 3: Generate PLACE Records ─────────────────────────────────
print('Generating Phase 3 (PLACE) records for completed candidates...\n')

# Only completed candidates proceed to placement
merged12     = phase1_df.merge(phase2_df, on='CandidateID')
completed_df = merged12[merged12['Completed'] == 1].copy()
print(f'Candidates proceeding to placement: {len(completed_df):,} / {len(merged12):,} '
      f'({len(completed_df)/len(merged12):.1%})')

PLACE_EMPLOYERS = {
    '49-2022.00': ['Charter Communications', 'US Cellular', 'Intrado',
                   'Avacend Corporation', 'Leidos', 'General Dynamics IT'],
    '49-9051.00': ['Pike Corporation', 'NPL Construction Co.', 'Q3 Contracting Inc.',
                   'Davey Tree Expert Co.', 'Altec Industries', 'PDS Tech Inc.'],
    '51-8013.00': ['Exelon Corporation', 'BGIS North America', 'Exide Technologies',
                   'Veolia North America', 'General Dynamics IT'],
    '51-8031.00': ['Veolia North America', 'BGIS North America',
                   'NPL Construction Co.', 'Exelon Corporation'],
    '51-8092.00': ['Exelon Corporation', 'General Dynamics IT',
                   'Veolia North America', 'Leidos', 'GPAC'],
    '43-5041.00': ['Exelon Corporation', 'US Cellular',
                   'Charter Communications', 'Veolia North America'],
}

phase3_records = []
placement_counter = 1

for _, row in completed_df.iterrows():
    soc = row['SOC_Code']
    cfg = SOC_CONFIG[soc]

    placement_id = f'PLC-{placement_counter:05d}'
    placement_counter += 1

    # Employment type by SOC
    if soc == '49-2022.00':
        emp_type = np.random.choice(['Full-Time', 'Part-Time', 'Contractor'], p=[0.86, 0.04, 0.10])
    elif soc == '49-9051.00':
        emp_type = np.random.choice(['Full-Time', 'Part-Time'], p=[0.93, 0.07])
    else:
        emp_type = np.random.choice(['Full-Time', 'Part-Time'], p=[0.92, 0.08])

    union = bernoulli(cfg['union_rate'])

    # Hire date = training end + realistic placement lag
    end_dt    = datetime.strptime(row['EndDate'], '%Y-%m-%d')
    lag_days  = np.random.randint(7, 91)
    hire_date = end_dt + timedelta(days=lag_days)

    # Role compliance requirements
    req_cdl    = bernoulli(min(cfg['cdl_rate'] * 1.1, 1.0))
    req_dot    = int(req_cdl)
    req_osha10 = bernoulli(min(cfg['osha10_rate'] * 1.1, 1.0))
    req_cpr    = bernoulli(min(cfg['cpr_rate'] * 1.1, 1.0))

    # Pre-hire verification — conditioned on candidate status
    mvr_ver  = bernoulli(0.92 if row['HasValidLicense'] else 0.05)
    drug_ver = bernoulli(0.96 if row['CanPassDrugScreen'] else 0.08)
    bg_ver   = bernoulli(0.94 if row['CanPassBackgroundCheck'] else 0.05)
    cert_ver = bernoulli(0.88 if row['CertificationsEarned'] > 0 else 0.10)

    # Orientation
    safety_orient_date = hire_date + timedelta(days=np.random.randint(0, 4))
    loto_comp = bernoulli(cfg['loto_rate'])
    ppe_fit   = bernoulli(0.95 if soc != '43-5041.00' else 0.70)

    # Apprenticeship registration
    appr_reg = bernoulli(
        0.85 if (row['ApprenticeshipInterest'] and union) else
        0.40 if row['ApprenticeshipInterest'] else 0.05
    )

    phase3_records.append({
        'CandidateID':                   row['CandidateID'],
        'PlacementID':                   placement_id,
        'EmployerName':                  np.random.choice(PLACE_EMPLOYERS[soc]),
        'JobFamily':                     row['JobFamily'],
        'EmploymentType':                emp_type,
        'UnionStatus':                   int(union),
        'HireDate':                      hire_date.strftime('%Y-%m-%d'),
        'RoleRequires_CDL':              int(req_cdl),
        'RoleRequires_DOTMedical':       req_dot,
        'RoleRequires_OSHA10':           int(req_osha10),
        'RoleRequires_CPR':              int(req_cpr),
        'PreHire_Verified_MVR':          int(mvr_ver),
        'PreHire_Verified_DrugScreen':   int(drug_ver),
        'PreHire_Verified_Background':   int(bg_ver),
        'PreHire_Verified_Certificates': int(cert_ver),
        'Orientation_SiteSafety_Date':   safety_orient_date.strftime('%Y-%m-%d'),
        'Orientation_LOTO_Completed':    int(loto_comp),
        'Orientation_PPE_Fitted':        int(ppe_fit),
        'Apprenticeship_Registered':     int(appr_reg),
        'Sorvex360PI_ScoreAtHire':       row['Sorvex360PI_Score_AtCompletion'],
    })

phase3_df = pd.DataFrame(phase3_records)
print(f'✅ Phase 3 complete: {len(phase3_df):,} placement records')
print(f'Union rate: {phase3_df["UnionStatus"].mean():.1%}')
print(f'Employment type:\n{phase3_df["EmploymentType"].value_counts()}')

# ── Cell 7 — Phase 4: Generate POST-HIRE OUTCOMES ────────────────────────────
# Signal design:
#   Tenure_1Year        → driven by PI Score + age + union status
#   OSHA_Incident       → driven by SafetyCommitmentScore + SOC base rate
#   PromotionWithin24M  → driven by ManagerFitFeedback + training hours + attendance
#   ManagerFitFeedback  → driven by PI Score + attendance (intermediate, NOT a target feature)
#
# NOTE: ManagerFitFeedback_Score is an outcome variable generated here in Phase 4.
# Do NOT use it as a predictor in your ML models — it would be data leakage.

print('Generating Phase 4 (POST-HIRE OUTCOMES) records...\n')

merged123 = completed_df.merge(phase3_df, on='CandidateID')

phase4_records = []

for _, row in merged123.iterrows():
    soc          = row['SOC_Code']
    cfg          = SOC_CONFIG[soc]
    pi           = row['Sorvex360PI_ScoreAtHire']
    age          = row['Age']
    union        = row['UnionStatus']
    safety_score = row['SafetyCommitmentScore']   # 1-5 scale from Phase 2
    train_hrs    = row['TotalTrainingHours']
    att          = row['AttendanceRate']

    # ── Tenure ───────────────────────────────────────────────────────────────
    # Base by SOC type, boosted by union, scaled by PI Score
    base_tenure_yr = cfg['base_tenure_yr']
    if union:
        base_tenure_yr *= 1.35
    pi_mult     = 0.8 + (pi / 100) * 0.4     # PI 0→0.80x, PI 100→1.20x
    tenure_yr   = max(0.1, np.random.weibull(1.2) * base_tenure_yr * pi_mult)
    tenure_days = int(clamp(round(tenure_yr * 365), 1, 7300))

    t90d = int(tenure_days >= 90)
    t6mo = int(tenure_days >= 180)
    # Tenure_1Year: combination of actual days AND age-banded probability (most realistic)
    t1yr_base  = tenure_1yr_rate(age) * pi_mult
    t1yr_days  = int(tenure_days >= 365)
    t1yr       = int(t1yr_days and bernoulli(min(t1yr_base, 1.0)))

    if tenure_days > 1000:
        status = np.random.choice(['Active', 'Separated', 'On Leave'], p=[0.65, 0.28, 0.07])
    else:
        status = np.random.choice(['Active', 'Separated'], p=[0.40, 0.60])

    # ── Safety (OSHA Recordable Incident) ────────────────────────────────────
    # Higher SafetyCommitmentScore → lower incident rate
    safety_score_norm = (5 - safety_score) / 4     # invert: higher score = lower risk
    soc_base_rate     = cfg['base_safety_rate']
    base_osha_prob    = safety_score_norm * soc_base_rate * 3
    exp_noise         = 0.015 if tenure_days < 365 else 0.0   # new workers slightly higher risk
    rand_noise        = np.random.normal(0, 0.01)             # tightened noise
    osha_rate         = clamp(base_osha_prob + exp_noise + rand_noise, 0.005, 0.50)
    osha_incident     = int(bernoulli(osha_rate))

    total_safety      = int(np.random.poisson(2.1 if osha_incident else 0.2))
    total_violations  = int(np.random.poisson(0.5 * (1 - safety_score / 5)))
    total_preventable = int(min(total_safety, np.random.binomial(max(total_safety, 1), 0.30)))

    # ── Attendance / Absences ─────────────────────────────────────────────────
    absence_lambda  = 3.0 if att > 0.90 else 5.0 if att > 0.80 else 9.0
    unscheduled_abs = int(np.random.poisson(absence_lambda))

    # Probation — strongly tied to PI Score
    probation = int(bernoulli(0.60 if pi < 55 else 0.22 if pi < 70 else 0.08))

    # ── Drug Screens ──────────────────────────────────────────────────────────
    screens_lambda = 2.5 if row['CDL_Status'] else 1.2
    screens_taken  = int(np.random.poisson(screens_lambda))
    screens_passed = sum(bernoulli(0.975) for _ in range(screens_taken))

    # ── Recertification ───────────────────────────────────────────────────────
    recert_safety = int(bernoulli(0.88 if safety_score > 3.5 else 0.62))
    recert_cpr    = int(bernoulli(0.91))

    # ── Manager Feedback Score (Phase 4 intermediate — NOT a predictor feature) ──
    if pi > 80:   mgr_mean = 3.9
    elif pi > 70: mgr_mean = 3.4
    elif pi > 60: mgr_mean = 2.9
    else:         mgr_mean = 2.3
    att_bonus = (att - 0.78) * 1.5
    mgr_score = round(clamp(np.random.normal(mgr_mean + att_bonus, 0.35), 1, 5), 2)

    # ── Promotion ─────────────────────────────────────────────────────────────
    # Signal: mgr_score (65%) + training hours (25%) + attendance (10%)
    # Must have >= 1 year tenure to be eligible
    mgr_norm   = (mgr_score - 1) / 4
    train_norm = clamp((train_hrs - 2000) / 8000, 0, 1)
    promo_score = mgr_norm * 0.65 + train_norm * 0.25 + (att - 0.78) * 0.10
    promo_noise = np.random.normal(0, 0.03)           # tightened from 0.04
    tenure_mult = 1.0 if tenure_days >= 365 else 0.05
    promo_prob  = clamp((promo_score * 0.43 + promo_noise) * tenure_mult, 0.01, 0.80)
    promotion   = int(bernoulli(promo_prob))

    # ── Time to Competency ────────────────────────────────────────────────────
    comp_months = round(clamp(
        np.random.lognormal(np.log(train_hrs / 160), 0.4), 1, 60
    ), 1)

    phase4_records.append({
        'CandidateID':                  row['CandidateID'],
        'PlacementID':                  row['PlacementID'],
        'TenureInDays':                 tenure_days,
        'Tenure_90Day':                 t90d,
        'Tenure_6Month':                t6mo,
        'Tenure_1Year':                 t1yr,
        'EmploymentStatus':             status,
        'OSHA_Recordable_Incident':     osha_incident,
        'TotalSafetyIncidents':         total_safety,
        'TotalComplianceViolations':    total_violations,
        'TotalPreventableAccidents':    total_preventable,
        'UnscheduledAbsences':          unscheduled_abs,
        'ProbationStatus':              probation,
        'RandomDrugScreens_Taken':      screens_taken,
        'RandomDrugScreens_Passed':     screens_passed,
        'Recert_Safety_PassedFirstTry': recert_safety,
        'Recert_CPR_PassedFirstTry':    recert_cpr,
        'ManagerFitFeedback_Score':     mgr_score,   # Phase 4 only — do not use as ML feature
        'PromotionWithin24Months':      promotion,
        'Time_To_Competency_Months':    comp_months,
    })

phase4_df = pd.DataFrame(phase4_records)
print(f'✅ Phase 4 complete: {len(phase4_df):,} outcome records')
print(f'\nOutcome rates:')
print(f'  Tenure_1Year:           {phase4_df["Tenure_1Year"].mean():.1%}')
print(f'  OSHA incident:          {phase4_df["OSHA_Recordable_Incident"].mean():.2%}')
print(f'  Promotion within 24mo:  {phase4_df["PromotionWithin24Months"].mean():.1%}')
print(f'  Probation:              {phase4_df["ProbationStatus"].mean():.1%}')
print(f'  Avg ManagerFeedback:    {phase4_df["ManagerFitFeedback_Score"].mean():.3f}')
print(f'\nOSHA rate by SOC:')
print(phase4_df.groupby(merged123['SOC_Code'])['OSHA_Recordable_Incident'].mean().round(3))

# ── Cell 8 — Validation ───────────────────────────────────────────────────────
print('=== SORVEX 360 VALIDATION REPORT ===')

# Blueprint target checks
checks = [
    ('PI Score mean',       phase1_df['Sorvex360PI_Score'].mean(),              60,   80  ),
    ('Completion rate',     phase2_df['Completed'].mean(),                       0.45, 0.65),
    ('Union rate',          phase3_df['UnionStatus'].mean(),                     0.15, 0.28),
    ('Tenure_1Year rate',   phase4_df['Tenure_1Year'].mean(),                    0.60, 0.85),
    ('OSHA incident rate',  phase4_df['OSHA_Recordable_Incident'].mean(),        0.01, 0.08),
    ('Promotion rate',      phase4_df['PromotionWithin24Months'].mean(),         0.15, 0.28),
    ('Probation rate',      phase4_df['ProbationStatus'].mean(),                 0.15, 0.35),
    ('Manager score mean',  phase4_df['ManagerFitFeedback_Score'].mean(),        2.7,  3.4 ),
]

all_pass = True
for name, val, lo, hi in checks:
    ok = lo <= val <= hi
    if not ok: all_pass = False
    icon = '✅' if ok else '❌'
    print(f'{icon} {name}: {val:.3f} (target {lo:.2f}–{hi:.2f})')

# Correlation checks — these are the ML signals we care about
print('\n=== KEY CORRELATION CHECKS ===')
master = (
    phase1_df
    .merge(phase2_df, on='CandidateID')
    .merge(phase3_df, on='CandidateID', how='left')
    .merge(phase4_df, on='CandidateID', how='left')
)

corr_checks = [
    ('CognitiveScore → Sorvex360PI_Score',         'CognitiveScore',       'Sorvex360PI_Score',           0.55),
    ('Age → LongestJobTenure',                      'Age',                  'LongestJobTenure',            0.30),
    ('Sorvex360PI_ScoreAtHire → Tenure_1Year',      'Sorvex360PI_ScoreAtHire','Tenure_1Year',             0.20),
    ('SafetyCommitmentScore → OSHA_Incident (neg)', 'SafetyCommitmentScore','OSHA_Recordable_Incident',  -0.15),
    ('AttendanceRate → PromotionWithin24Months',    'AttendanceRate',       'PromotionWithin24Months',     0.15),
]
for name, c1, c2, threshold in corr_checks:
    cols = [c1, c2]
    available = [c for c in cols if c in master.columns]
    if len(available) < 2:
        print(f'⚠️  {name}: column missing, skipped')
        continue
    r = master[available].corr().iloc[0, 1]
    if threshold >= 0:
        ok = r >= threshold
    else:
        ok = r <= threshold
    icon = '✅' if ok else '⚠️ '
    print(f'{icon} {name}: r={r:.3f} (threshold {threshold:+.2f})')

print(f'\n{"✅ All checks passed!" if all_pass else "⚠️  Some checks outside target — review before training"}')
print(f'Total placed candidates available for ML: {len(master):,}')

# ── Cell 9 — Save Phase CSVs + Master Flat File ───────────────────────────────
print('Saving output files...\n')

p1_cols = [
    'CandidateID', 'SOC_Code', 'JobFamily', 'Age', 'Gender', 'State',
    'PrimaryLanguage', 'VeteranStatus', 'EducationLevel', 'CDL_Status',
    'OSHA10_Status', 'CPR_Status', 'CognitiveScore', 'SimulationScore',
    'BehavioralScore', 'SituationalScore', 'HasPriorTradeExperience',
    'MostRecentIndustry', 'LongestJobTenure', 'TrainingSource',
    'ApprenticeshipInterest', 'HasValidLicense', 'CanPassDrugScreen',
    'CanPassBackgroundCheck', 'SourceOfCandidate', 'Sorvex360PI_Score',
]

p2_cols = [
    'CandidateID', 'TrainingProvider', 'StartDate', 'EndDate',
    'TotalTrainingHours', 'AttendanceRate', 'PassedRequiredModules',
    'CertificationsEarned', 'SimulationPerformance', 'TeamworkScore',
    'SafetyCommitmentScore', 'ReliabilityScore', 'PhysicalTestResult',
    'PoleClimbResult', 'Lift50lbsTest', 'ColorBlindness', 'Completed',
    'DropoutReason', 'Sorvex360PI_Score_AtCompletion', 'ReadinessDelta',
]

p3_cols = [
    'CandidateID', 'PlacementID', 'EmployerName', 'JobFamily',
    'EmploymentType', 'UnionStatus', 'HireDate', 'RoleRequires_CDL',
    'RoleRequires_DOTMedical', 'RoleRequires_OSHA10', 'RoleRequires_CPR',
    'PreHire_Verified_MVR', 'PreHire_Verified_DrugScreen',
    'PreHire_Verified_Background', 'PreHire_Verified_Certificates',
    'Orientation_SiteSafety_Date', 'Orientation_LOTO_Completed',
    'Orientation_PPE_Fitted', 'Apprenticeship_Registered',
    'Sorvex360PI_ScoreAtHire',
]

p4_cols = [
    'CandidateID', 'PlacementID', 'TenureInDays', 'Tenure_90Day',
    'Tenure_6Month', 'Tenure_1Year', 'EmploymentStatus',
    'OSHA_Recordable_Incident', 'TotalSafetyIncidents',
    'TotalComplianceViolations', 'TotalPreventableAccidents',
    'UnscheduledAbsences', 'ProbationStatus', 'RandomDrugScreens_Taken',
    'RandomDrugScreens_Passed', 'Recert_Safety_PassedFirstTry',
    'Recert_CPR_PassedFirstTry', 'ManagerFitFeedback_Score',
    'PromotionWithin24Months', 'Time_To_Competency_Months',
]

# Save individual phase files
for phase_name, cols, df_src in [
    ('Phase1_Predict',  p1_cols, phase1_df),
    ('Phase2_Prepare',  p2_cols, phase2_df),
    ('Phase3_Place',    p3_cols, phase3_df),
    ('Phase4_PostHire', p4_cols, phase4_df),
]:
    avail  = [c for c in cols if c in df_src.columns]
    df_out = df_src[avail].copy()
    path   = OUTPUT_DIR / f'Sorvex360_{phase_name}.csv'
    df_out.to_csv(path, index=False)
    print(f'  ✅ {phase_name}: {len(df_out):,} rows × {len(avail)} cols → {path.name}')

# Save master flat file — this is what you load in the Vertex notebook
# Contains only PLACED candidates (completed training + hired)
master_path = OUTPUT_DIR / 'Sorvex360_Master_Flat.csv'
master.to_csv(master_path, index=False)
print(f'  ✅ Master flat file: {len(master):,} rows × {len(master.columns)} cols → Sorvex360_Master_Flat.csv')
print(f'\n⚠️  Reminder: Do NOT use ManagerFitFeedback_Score as a predictor in ML models.')
print(f'   It is a Phase 4 outcome variable generated alongside your targets.')

# ── Cell 10 — Validation Charts ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Sorvex 360 — Synthetic Dataset Validation', fontsize=14, fontweight='bold')

# PI Score distribution
phase1_df['Sorvex360PI_Score'].hist(bins=40, ax=axes[0,0], color='steelblue', edgecolor='white')
axes[0,0].set_title('PI Score Distribution')
axes[0,0].set_xlabel('Score (0–100)')
axes[0,0].axvline(phase1_df['Sorvex360PI_Score'].mean(), color='red', linestyle='--', label='mean')
axes[0,0].legend()

# SOC distribution
soc_labels = {k: v['label'].split('/')[0][:20] for k, v in SOC_CONFIG.items()}
phase1_df['SOC_Code'].map(soc_labels).value_counts().plot(
    kind='barh', ax=axes[0,1], color='coral')
axes[0,1].set_title('Records by SOC')

# Tenure distribution (placed candidates)
(phase4_df['TenureInDays'] / 365).hist(bins=40, ax=axes[0,2], color='mediumseagreen', edgecolor='white')
axes[0,2].set_title('Tenure Distribution (Years)')
axes[0,2].set_xlabel('Years')

# Attendance rate distribution
phase2_df['AttendanceRate'].hist(bins=30, ax=axes[1,0], color='mediumpurple', edgecolor='white')
axes[1,0].set_title('Attendance Rate Distribution')
axes[1,0].set_xlabel('Attendance Rate')

# OSHA rate by SOC
osha_by_soc = phase4_df.groupby(merged123['SOC_Code'])['OSHA_Recordable_Incident'].mean()
osha_by_soc.index = osha_by_soc.index.map(soc_labels)
osha_by_soc.plot(kind='barh', ax=axes[1,1], color='tomato')
axes[1,1].set_title('OSHA Incident Rate by SOC')
axes[1,1].set_xlabel('Rate')

# Outcome summary
outcomes = {
    'Tenure 1yr': phase4_df['Tenure_1Year'].mean(),
    'Promotion': phase4_df['PromotionWithin24Months'].mean(),
    'Probation': phase4_df['ProbationStatus'].mean(),
    'OSHA': phase4_df['OSHA_Recordable_Incident'].mean(),
}
axes[1,2].bar(outcomes.keys(), outcomes.values(), color=['steelblue','mediumseagreen','tomato','orange'])
axes[1,2].set_title('Phase 4 Outcome Rates')
axes[1,2].set_ylabel('Rate')
for i, (k, v) in enumerate(outcomes.items()):
    axes[1,2].text(i, v + 0.005, f'{v:.1%}', ha='center', fontsize=9)

plt.tight_layout()
chart_path = OUTPUT_DIR / 'validation_charts.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Charts saved to {chart_path}')

# ── Cell 11 — Final Summary + Download ───────────────────────────────────────
print('=' * 55)
print('SORVEX 360 — SYNTHETIC PIPELINE V3 COMPLETE')
print('=' * 55)
print(f'Total candidates generated:     {N:,}')
print(f'Completed training:             {phase2_df["Completed"].sum():,} ({phase2_df["Completed"].mean():.1%})')
print(f'Placed (Phase 3 + 4 records):   {len(phase3_df):,}')
print(f'Master flat file rows:          {len(master):,}')
print(f'Master flat file columns:       {len(master.columns)}')
print()
print('Output files:')
import os
for f in sorted(os.listdir(OUTPUT_DIR)):
    size_kb = (OUTPUT_DIR / f).stat().st_size / 1024
    print(f'  {f:<45} {size_kb:>6.0f} KB')
print()
print('ML targets in Master_Flat.csv:')
print('  Tenure_1Year              — retention (binary)')
print('  OSHA_Recordable_Incident  — safety risk (binary)')
print('  PromotionWithin24Months   — advancement (binary)')
print()
print('DO NOT use as ML features (Phase 4 leakage):')
print('  ManagerFitFeedback_Score, TenureInDays,')
print('  UnscheduledAbsences, ProbationStatus,')
print('  TotalSafetyIncidents, TotalComplianceViolations')

# Download all files
try:
    from google.colab import files
    print('\nDownloading all output files...')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        files.download(str(OUTPUT_DIR / f))
    print('✅ All downloads triggered.')
except ImportError:
    print(f'\n(Not running in Colab — files saved to {OUTPUT_DIR})')
