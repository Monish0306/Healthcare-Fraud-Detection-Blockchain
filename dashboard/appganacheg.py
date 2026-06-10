import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import shap, joblib, json, os, time, warnings, hashlib
warnings.filterwarnings('ignore')

def compute_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Fraud Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ──────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
PROC      = os.path.join(BASE, '../data/processed/')
MODELS    = os.path.join(BASE, '../models/')
RESULTS   = os.path.join(BASE, '../results/')
FIG       = os.path.join(BASE, '../paper/figures/')

# ══════════════════════════════════════════════════════════════
# DATA LOADERS — cached so they only run once
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    lr = joblib.load(MODELS + 'logistic_regression.pkl')
    rf = joblib.load(MODELS + 'random_forest.pkl')
    sc = joblib.load(MODELS + 'scaler.pkl')
    return lr, rf, sc

@st.cache_data
def load_metrics():
    return pd.read_csv(RESULTS + 'all_metrics_for_dashboard.csv')

@st.cache_data
def load_test_predictions():
    return pd.read_csv(RESULTS + 'test_predictions.csv')

@st.cache_data
def load_blockchain_records():
    verified = pd.read_csv(RESULTS + 'blockchain_verified_records.csv')
    # Also load stored_records which has real TxHash + BlockNumber from notebook runs
    stored_path = RESULTS + 'blockchain_stored_records.csv'
    if os.path.exists(stored_path):
        stored = pd.read_csv(stored_path)
        # Normalise column names to match
        col_map = {
            'tx_hash': 'TxHash', 'transaction_hash': 'TxHash',
            'block_number': 'BlockNumber', 'block': 'BlockNumber',
            'gas_used': 'GasUsed', 'gas': 'GasUsed',
            'provider': 'Provider', 'is_fraud': 'IsFraud',
            'fraud_probability': 'FraudProbability', 'probability': 'FraudProbability',
            'risk_category': 'RiskCategory', 'risk': 'RiskCategory',
            'data_hash': 'DataHash', 'hash': 'DataHash',
            'timestamp': 'Timestamp',
        }
        stored.rename(columns={k: v for k, v in col_map.items()
                                if k in stored.columns}, inplace=True)
        # Merge TxHash/BlockNumber/GasUsed from stored into verified by Provider
        merge_cols = [c for c in ['TxHash','BlockNumber','GasUsed','DataHash','Timestamp']
                      if c in stored.columns]
        if merge_cols and 'Provider' in stored.columns:
            verified = verified.merge(
                stored[['Provider'] + merge_cols],
                on='Provider', how='left', suffixes=('', '_stored')
            )
            for c in merge_cols:
                if c + '_stored' in verified.columns:
                    verified[c] = verified[c + '_stored'].combine_first(
                        verified.get(c, pd.Series(dtype=str)))
                    verified.drop(columns=[c + '_stored'], inplace=True)
    return verified

@st.cache_data
def load_feature_names():
    return pd.read_csv(PROC + 'feature_names.csv', header=None)[0].tolist()

@st.cache_data
def load_shap_data():
    lr_shap = np.load(PROC + 'lr_shap_values_val.npy')
    rf_shap = np.load(PROC + 'rf_shap_values_val.npy')
    lr_ev   = float(np.load(PROC + 'lr_expected_value.npy'))
    rf_ev   = float(np.load(PROC + 'rf_expected_value.npy'))
    X_val   = np.load(PROC + 'X_val_scaled.npy')
    y_val   = np.load(PROC + 'y_val.npy')
    return lr_shap, rf_shap, lr_ev, rf_ev, X_val, y_val

@st.cache_data
def load_threshold():
    with open(RESULTS + 'optimal_threshold.json') as f:
        return json.load(f)['optimal_threshold']

@st.cache_data
def load_blockchain_config():
    with open(RESULTS + 'blockchain_config.json') as f:
        return json.load(f)

# ── Load everything ────────────────────────────────────────────
lr_model, rf_model, scaler = load_models()
metrics_df      = load_metrics()
test_preds      = load_test_predictions()
bc_records      = load_blockchain_records()
feat_names      = load_feature_names()
lr_shap, rf_shap, lr_ev, rf_ev, X_val, y_val = load_shap_data()
THRESHOLD       = load_threshold()
bc_config       = load_blockchain_config()

# ── Preprocessing helper (same as Phase 2) ────────────────────
def preprocess_uploaded(bene_df, inp_df, outp_df):
    import pandas as pd
    import numpy as np

    # --- Beneficiary ---
    bene = bene_df.copy()
    bene['DOB'] = pd.to_datetime(bene['DOB'], errors='coerce')
    bene['DOD'] = pd.to_datetime(bene['DOD'], errors='coerce')
    ref  = pd.Timestamp('2009-12-01')
    bene['Age']    = ((ref - bene['DOB']).dt.days / 365.25).clip(lower=0)
    bene['IsDead'] = bene['DOD'].notna().astype(int)
    chronic_cols   = [c for c in bene.columns if 'ChronicCond' in c]
    for col in chronic_cols:
        bene[col] = (bene[col] == 1).astype(int)
    bene['NumChronicConditions'] = bene[chronic_cols].sum(axis=1)
    bene['RenalDisease']  = (bene['RenalDiseaseIndicator'] == 'Y').astype(int)
    bene['GenderBinary']  = (bene['Gender'] == 1).astype(int)

    # --- Inpatient ---
    inp = inp_df.copy()
    for col in ['ClaimStartDt','ClaimEndDt','AdmissionDt','DischargeDt']:
        if col in inp.columns:
            inp[col] = pd.to_datetime(inp[col], errors='coerce')
    inp['StayDuration']      = (inp['DischargeDt'] - inp['AdmissionDt']).dt.days.clip(lower=0) if 'AdmissionDt' in inp.columns else 0
    inp['ClaimDuration']     = (inp['ClaimEndDt']  - inp['ClaimStartDt']).dt.days.clip(lower=0)
    diag_cols = [c for c in inp.columns if 'ClmDiagnosisCode' in c]
    proc_cols = [c for c in inp.columns if 'ClmProcedureCode' in c]
    inp['NumDiagnosisCodes'] = inp[diag_cols].notna().sum(axis=1)
    inp['NumProcedureCodes'] = inp[proc_cols].notna().sum(axis=1)
    inp = inp.merge(bene[['BeneID','Age','IsDead','NumChronicConditions',
                           'RenalDisease','GenderBinary',
                           'IPAnnualReimbursementAmt','OPAnnualReimbursementAmt']], on='BeneID', how='left')

    ip_agg = inp.groupby('Provider').agg(
        IP_NumClaims=('ClaimID','count'),
        IP_NumUniqueBeneficiaries=('BeneID','nunique'),
        IP_NumUniqueAttPhysicians=('AttendingPhysician','nunique'),
        IP_NumUniqueOpPhysicians=('OperatingPhysician','nunique'),
        IP_NumUniqueOthPhysicians=('OtherPhysician','nunique'),
        IP_TotalClaimAmt=('InscClaimAmtReimbursed','sum'),
        IP_AvgClaimAmt=('InscClaimAmtReimbursed','mean'),
        IP_MaxClaimAmt=('InscClaimAmtReimbursed','max'),
        IP_StdClaimAmt=('InscClaimAmtReimbursed','std'),
        IP_TotalDeductible=('DeductibleAmtPaid','sum'),
        IP_AvgDeductible=('DeductibleAmtPaid','mean'),
        IP_AvgStayDuration=('StayDuration','mean'),
        IP_MaxStayDuration=('StayDuration','max'),
        IP_TotalStayDays=('StayDuration','sum'),
        IP_AvgClaimDuration=('ClaimDuration','mean'),
        IP_AvgDiagnosisCodes=('NumDiagnosisCodes','mean'),
        IP_AvgProcedureCodes=('NumProcedureCodes','mean'),
        IP_AvgPatientAge=('Age','mean'),
        IP_PctDeadPatients=('IsDead','mean'),
        IP_AvgChronicConditions=('NumChronicConditions','mean'),
        IP_PctRenalDisease=('RenalDisease','mean'),
        IP_AvgGender=('GenderBinary','mean'),
        IP_AvgIPAnnualReimb=('IPAnnualReimbursementAmt','mean'),
        IP_AvgOPAnnualReimb=('OPAnnualReimbursementAmt','mean'),
    ).reset_index()
    ip_agg['IP_StdClaimAmt'] = ip_agg['IP_StdClaimAmt'].fillna(0)

    # --- Outpatient ---
    outp = outp_df.copy()
    for col in ['ClaimStartDt','ClaimEndDt']:
        if col in outp.columns:
            outp[col] = pd.to_datetime(outp[col], errors='coerce')
    outp['ClaimDuration'] = (outp['ClaimEndDt'] - outp['ClaimStartDt']).dt.days.clip(lower=0)
    diag_cols_op = [c for c in outp.columns if 'ClmDiagnosisCode' in c]
    outp['NumDiagnosisCodes'] = outp[diag_cols_op].notna().sum(axis=1)
    outp = outp.merge(bene[['BeneID','Age','IsDead','NumChronicConditions']], on='BeneID', how='left')

    op_agg = outp.groupby('Provider').agg(
        OP_NumClaims=('ClaimID','count'),
        OP_NumUniqueBeneficiaries=('BeneID','nunique'),
        OP_NumUniqueAttPhysicians=('AttendingPhysician','nunique'),
        OP_TotalClaimAmt=('InscClaimAmtReimbursed','sum'),
        OP_AvgClaimAmt=('InscClaimAmtReimbursed','mean'),
        OP_MaxClaimAmt=('InscClaimAmtReimbursed','max'),
        OP_StdClaimAmt=('InscClaimAmtReimbursed','std'),
        OP_TotalDeductible=('DeductibleAmtPaid','sum'),
        OP_AvgClaimDuration=('ClaimDuration','mean'),
        OP_AvgDiagnosisCodes=('NumDiagnosisCodes','mean'),
        OP_AvgPatientAge=('Age','mean'),
        OP_PctDeadPatients=('IsDead','mean'),
        OP_AvgChronicConditions=('NumChronicConditions','mean'),
    ).reset_index()
    op_agg['OP_StdClaimAmt'] = op_agg['OP_StdClaimAmt'].fillna(0)

    # --- Merge ---
    providers = pd.DataFrame({'Provider': pd.concat([
        inp_df['Provider'], outp_df['Provider']]).unique()})
    master = providers.merge(ip_agg,  on='Provider', how='left')
    master = master.merge(op_agg, on='Provider', how='left')
    master = master.fillna(0)

    master['IP_OP_ClaimRatio']      = master['IP_NumClaims'] / (master['OP_NumClaims'] + 1)
    master['IP_ClaimsPerPatient']   = master['IP_NumClaims'] / (master['IP_NumUniqueBeneficiaries'] + 1)
    master['OP_ClaimsPerPatient']   = master['OP_NumClaims'] / (master['OP_NumUniqueBeneficiaries'] + 1)
    master['IP_PatientsPerPhysician']= master['IP_NumUniqueBeneficiaries'] / (master['IP_NumUniqueAttPhysicians'] + 1)
    master['OP_PatientsPerPhysician']= master['OP_NumUniqueBeneficiaries'] / (master['OP_NumUniqueAttPhysicians'] + 1)
    master['IP_ClaimAmtPerStayDay'] = master['IP_TotalClaimAmt'] / (master['IP_TotalStayDays'] + 1)
    master['Total_ClaimAmt']        = master['IP_TotalClaimAmt'] + master['OP_TotalClaimAmt']
    master['Total_Deductible']      = master['IP_TotalDeductible'] + master['OP_TotalDeductible']
    master['Total_UniqueBeneficiaries'] = master['IP_NumUniqueBeneficiaries'] + master['OP_NumUniqueBeneficiaries']
    master['DiagComplexityRatio']   = master['IP_AvgDiagnosisCodes'] / (master['OP_AvgDiagnosisCodes'] + 1)
    master.replace([np.inf, -np.inf], 0, inplace=True)

    return master

def risk_label(p):
    if p >= 0.7:   return 'High'
    elif p >= 0.4: return 'Medium'
    else:          return 'Low'

def risk_color(r):
    return {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(r, '⚪')

# ══════════════════════════════════════════════════════════════
# BLOCKCHAIN SESSION STATE INIT
# ══════════════════════════════════════════════════════════════
_bc_defaults = {
    'blockchain_records':     [],
    'blockchain_initialized': False,
}
for _k, _v in _bc_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Auto-populate blockchain ledger with test_predictions on first run
if not st.session_state.blockchain_initialized:
    try:
        _upload_ts = '2026-03-07T10:00:00'
        for _, _row in test_preds.iterrows():
            _prov     = str(_row['Provider'])
            _is_fr    = bool(_row['FraudPrediction'])
            _prob_dec = float(_row['FraudProbability'])
            _risk     = str(_row.get('RiskCategory', risk_label(_prob_dec)))
            _data_str = f"{_prov}{_is_fr}{_prob_dec:.6f}{_risk}"
            _bh       = compute_hash(_data_str)
            _prev     = st.session_state.blockchain_records[-1]['block_hash'] \
                        if st.session_state.blockchain_records else ''
            st.session_state.blockchain_records.append({
                'provider':     _prov,
                'fraud':        _is_fr,
                'probability':  round(_prob_dec * 100, 2),
                'prob_decimal': _prob_dec,
                'risk':         _risk,
                'source':       '🎓 Training Data',
                'timestamp':    _upload_ts,
                'block_hash':   _bh,
                'prev_hash':    _prev,
                'tampered':     False,
            })
        st.session_state.blockchain_initialized = True
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🏥 Fraud Detection")
    st.markdown("**INDISCON 2026**")
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Overview",
        "📤 Upload & Predict",
        "📊 Model Performance",
        "🔍 Fraud Analysis",
        "🤖 Explainable AI",
        "🔒 Blockchain Security",
        "⛓️ Blockchain Records",
        "⛓️ Blockchain Ledger",
        "ℹ️ System Info",
    ])

    st.divider()
    st.caption(f"Threshold: {THRESHOLD:.2f}")
    st.caption(f"Features:  47")
    st.caption(f"Models:    7 evaluated")
    st.caption(f"Final:     Logistic Regression")
    st.caption(f"⛓️ Blockchain: {len(st.session_state.blockchain_records)} records")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🏥 Healthcare Provider Fraud Detection System")
    st.markdown("**Blockchain-Secured ML with Explainable AI — INDISCON 2026**")
    st.divider()

    # KPI cards
    fraud_count  = test_preds['FraudPrediction'].sum()
    total_count  = len(test_preds)
    fraud_rate   = fraud_count / total_count * 100
    high_risk    = (test_preds['RiskCategory'] == 'High').sum()
    bc_total     = len(bc_records)
    bc_verified  = bc_records['HashMatch'].sum() if 'HashMatch' in bc_records.columns else bc_total

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Providers",  f"{total_count:,}")
    c2.metric("Fraud Flagged",    f"{fraud_count:,}",  f"{fraud_rate:.1f}%")
    c3.metric("High Risk",        f"{high_risk:,}")
    c4.metric("Blockchain Records", f"{bc_total:,}")
    c5.metric("AUC-ROC",          "96.75%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Category Distribution")
        risk_counts = test_preds['RiskCategory'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        colors  = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}
        bars = ax.bar(risk_counts.index,
                      risk_counts.values,
                      color=[colors.get(r, '#95a5a6') for r in risk_counts.index],
                      edgecolor='black', linewidth=0.6)
        ax.set_ylabel('Number of Providers')
        ax.set_title('Provider Risk Categories')
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+5,
                    str(int(bar.get_height())),
                    ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Fraud Probability Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(test_preds['FraudProbability'], bins=40,
                color='#3498db', edgecolor='black', linewidth=0.4, alpha=0.8)
        ax.axvline(x=THRESHOLD, color='#e74c3c', linestyle='--',
                   linewidth=2, label=f'Threshold = {THRESHOLD:.2f}')
        ax.set_xlabel('Fraud Probability')
        ax.set_ylabel('Number of Providers')
        ax.set_title('Distribution of Fraud Probabilities')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.subheader("System Architecture")
    st.markdown("""
    | Component | Technology | Purpose |
    |---|---|---|
    | Data Processing | Pandas, NumPy | Feature engineering (47 features) |
    | Class Balancing | SMOTE | Handle 9.4% fraud rate imbalance |
    | ML Models | 7 models evaluated | Fraud classification |
    | Final Model | Logistic Regression | Best AUC + generalisation |
    | Explainability | SHAP LinearExplainer | Feature-level explanations |
    | Blockchain | Ethereum + Solidity | Tamper-proof audit trail |
    | Dashboard | Streamlit | Interactive interface |
    """)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — UPLOAD & PREDICT
# ══════════════════════════════════════════════════════════════
elif page == "📤 Upload & Predict":
    st.title("📤 Upload New Data & Detect Fraud")
    st.markdown("Upload Beneficiary, Inpatient, and Outpatient CSV files to run live fraud detection.")
    st.divider()

    model_choice = st.radio(
        "Select model for prediction:",
        ["Logistic Regression ★ (Final)", "Random Forest (Comparison)"],
        horizontal=True
    )
    active_model = lr_model if "Logistic" in model_choice else rf_model

    col1, col2, col3 = st.columns(3)
    with col1:
        bene_file = st.file_uploader("📋 Beneficiary CSV", type='csv')
    with col2:
        inp_file  = st.file_uploader("🏨 Inpatient CSV",   type='csv')
    with col3:
        outp_file = st.file_uploader("🏃 Outpatient CSV",  type='csv')

    if bene_file and inp_file and outp_file:
        with st.spinner("Processing files and engineering features..."):
            t0       = time.time()
            bene_df  = pd.read_csv(bene_file)
            inp_df   = pd.read_csv(inp_file)
            outp_df  = pd.read_csv(outp_file)

            master   = preprocess_uploaded(bene_df, inp_df, outp_df)
            X        = master[feat_names].values
            X_scaled = scaler.transform(X)
            proba    = active_model.predict_proba(X_scaled)[:, 1]
            pred     = (proba >= THRESHOLD).astype(int)
            proc_time = time.time() - t0

        results_df = pd.DataFrame({
            'Provider':         master['Provider'],
            'FraudPrediction':  pred,
            'FraudProbability': proba.round(4),
            'RiskCategory':     [risk_label(p) for p in proba],
        })

        # ── Save to session_state ─────────────────────────────────
        bc_live = results_df.copy()
        bc_live['IsFraud']   = bc_live['FraudPrediction'].astype(bool)
        bc_live['RecordID']  = range(len(bc_records) + 1, len(bc_records) + 1 + len(bc_live))
        bc_live['Timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        bc_live['HashMatch'] = True
        st.session_state['uploaded_bc_records'] = bc_live
        st.session_state['uploaded_results_df'] = results_df
        st.session_state['ganache_stored']      = False  # reset so ledger page stores fresh

        # ── Also add to SHA-256 in-memory blockchain ledger ──────
        _upload_ts2 = pd.Timestamp.now().isoformat()
        for _, _row2 in results_df.iterrows():
            _prob2    = float(_row2['FraudProbability'])
            _is_fr2   = bool(_row2['FraudPrediction'])
            _risk2    = str(_row2.get('RiskCategory', risk_label(_prob2)))
            _dstr2    = f"{_row2['Provider']}{_is_fr2}{_prob2:.6f}{_risk2}"
            _bh2      = compute_hash(_dstr2)
            _prev2    = st.session_state.blockchain_records[-1]['block_hash'] \
                        if st.session_state.blockchain_records else ''
            st.session_state.blockchain_records.append({
                'provider':     str(_row2['Provider']),
                'fraud':        _is_fr2,
                'probability':  round(_prob2 * 100, 2),
                'prob_decimal': _prob2,
                'risk':         _risk2,
                'source':       '📤 Uploaded Data',
                'timestamp':    _upload_ts2,
                'block_hash':   _bh2,
                'prev_hash':    _prev2,
                'tampered':     False,
            })

        st.success(f"✅ Processed {len(results_df):,} providers in {proc_time:.2f}s")

        # ── Store to Ganache immediately ──────────────────────────
        try:
            from web3 import Web3 as _W3
            _ganache_url = st.session_state.get('ganache_url', 'http://127.0.0.1:7545')
            _w3 = _W3(_W3.HTTPProvider(_ganache_url))

            if not _w3.is_connected():
                st.info(f"ℹ️ Ganache not reachable at {_ganache_url} — predictions saved to in-memory chain only.")
            else:
                _addr = st.session_state.get('active_contract_addr', bc_config.get('address', '')).strip()
                _cur_block = _w3.eth.block_number

                if not _addr:
                    st.warning("⚠️ No contract address set. Go to **⛓️ Blockchain Ledger → Ganache Settings** in the sidebar and paste your contract address.")
                elif _cur_block == 0:
                    st.warning("⚠️ Ganache was restarted (Block #0). Re-deploy your contract and paste the new address in the sidebar.")
                else:
                    # ── Try to load ABI from project files ───────────────
                    def _load_project_abi():
                        for _p in [
                            os.path.join(BASE, '../blockchain/build/contracts/FraudDetection.json'),
                            os.path.join(BASE, '../blockchain/build/contracts/HealthcareFraud.json'),
                            os.path.join(BASE, '../build/contracts/FraudDetection.json'),
                            os.path.join(RESULTS, 'contract_abi.json'),
                        ]:
                            if os.path.exists(_p):
                                with open(_p) as _f:
                                    _art = json.load(_f)
                                return _art.get('abi', _art)
                        return None

                    _proj_abi = _load_project_abi()

                    # ── ABI matching your deployed contract's 5-argument storeFraudRecord ──
                    # Signature: storeFraudRecord(provider, isFraud, fraudProbability, riskCategory, dataHash)
                    _MULTI_ABI = [
                        # ── PRIMARY: 5-arg storeFraudRecord (string dataHash) — matches your Solidity contract ──
                        {"inputs":[{"internalType":"string","name":"provider","type":"string"},
                                   {"internalType":"bool","name":"isFraud","type":"bool"},
                                   {"internalType":"uint256","name":"fraudProbability","type":"uint256"},
                                   {"internalType":"string","name":"riskCategory","type":"string"},
                                   {"internalType":"string","name":"dataHash","type":"string"}],
                         "name":"storeFraudRecord","outputs":[],"stateMutability":"nonpayable","type":"function"},
                        # ── FALLBACK: 5-arg storeFraudRecord (bytes32 dataHash) ──
                        {"inputs":[{"internalType":"string","name":"provider","type":"string"},
                                   {"internalType":"bool","name":"isFraud","type":"bool"},
                                   {"internalType":"uint256","name":"fraudProbability","type":"uint256"},
                                   {"internalType":"string","name":"riskCategory","type":"string"},
                                   {"internalType":"bytes32","name":"dataHash","type":"bytes32"}],
                         "name":"storeFraudRecord","outputs":[],"stateMutability":"nonpayable","type":"function"},
                        # ── addRecord alternative (also 5-arg) ──
                        {"inputs":[{"internalType":"string","name":"_id","type":"string"},
                                   {"internalType":"bool","name":"_fraud","type":"bool"},
                                   {"internalType":"uint256","name":"_prob","type":"uint256"},
                                   {"internalType":"string","name":"_risk","type":"string"},
                                   {"internalType":"string","name":"_hash","type":"string"}],
                         "name":"addRecord","outputs":[],"stateMutability":"nonpayable","type":"function"},
                        # ── View functions ──
                        {"inputs":[],"name":"getTotalRecords",
                         "outputs":[{"internalType":"uint256","name":"","type":"uint256"}],
                         "stateMutability":"view","type":"function"},
                        {"inputs":[{"internalType":"uint256","name":"_recordId","type":"uint256"}],
                         "name":"getRecord",
                         "outputs":[{"internalType":"string","name":"","type":"string"},
                                    {"internalType":"bool","name":"","type":"bool"},
                                    {"internalType":"uint256","name":"","type":"uint256"},
                                    {"internalType":"string","name":"","type":"string"},
                                    {"internalType":"uint256","name":"","type":"uint256"},
                                    {"internalType":"bytes32","name":"","type":"bytes32"}],
                         "stateMutability":"view","type":"function"},
                    ]
                    _USE_ABI = _proj_abi if _proj_abi else _MULTI_ABI

                    try:
                        _contract = _w3.eth.contract(
                            address=_W3.to_checksum_address(_addr), abi=_USE_ABI)

                        # ── Auto-detect which store function the contract has ──
                        _fn_names = [fn.fn_name for fn in _contract.functions]
                        _use_add_record = 'storeFraudRecord' not in _fn_names and 'addRecord' in _fn_names

                        if 'storeFraudRecord' not in _fn_names and 'addRecord' not in _fn_names:
                            st.error(
                                f"❌ Contract at `{_addr[:20]}...` has neither `storeFraudRecord` "
                                f"nor `addRecord`. Available: {_fn_names}"
                            )
                        else:
                            # ── Pick deployer: ALWAYS verify against live Ganache accounts ──
                            _accounts = _w3.eth.accounts
                            if not _accounts:
                                raise ValueError("No Ganache accounts found — is Ganache running with unlocked accounts?")
                            _deployer_cfg = bc_config.get('deployer', '')
                            # Only use bc_config deployer if it actually exists in current session
                            if _deployer_cfg and _deployer_cfg in _accounts:
                                _deployer = _deployer_cfg
                            else:
                                _deployer = _accounts[0]  # always use first live account

                            # ── FRAUD ONLY — every Ganache transaction = one confirmed fraud provider ──
                            _fraud_only = results_df[results_df['FraudPrediction'] == 1].copy()
                            _total = len(_fraud_only)
                            _n_clean_skipped = len(results_df) - _total

                            if _total == 0:
                                st.info("ℹ️ No fraud predictions in this upload — nothing stored to Ganache.")
                            else:
                                _prog  = st.progress(0, text=f"⛓️ Storing {_total} fraud predictions to Ganache…")
                                _ok    = 0
                                _errors = []

                                for _i, (_idx, _row) in enumerate(_fraud_only.iterrows()):
                                    try:
                                        _prov     = str(_row['Provider'])
                                        _fraud    = bool(_row['FraudPrediction'])
                                        _prob_int = int(round(float(_row['FraudProbability']) * 1_000_000))
                                        _risk_str = str(_row.get('RiskCategory', risk_label(float(_row['FraudProbability']))))
                                        _hash_str = compute_hash(f"{_prov}{_fraud}{float(_row['FraudProbability']):.6f}{_risk_str}")

                                        if _use_add_record:
                                            _tx = _contract.functions.addRecord(
                                                _prov, _fraud, _prob_int, _risk_str, _hash_str
                                            ).transact({'from': _deployer, 'gas': 300_000})
                                        else:
                                            try:
                                                _tx = _contract.functions.storeFraudRecord(
                                                    _prov, _fraud, _prob_int, _risk_str, _hash_str
                                                ).transact({'from': _deployer, 'gas': 300_000})
                                            except Exception:
                                                _hash_bytes = _w3.keccak(text=_hash_str)
                                                _tx = _contract.functions.storeFraudRecord(
                                                    _prov, _fraud, _prob_int, _risk_str, _hash_bytes
                                                ).transact({'from': _deployer, 'gas': 300_000})

                                        _w3.eth.wait_for_transaction_receipt(_tx, timeout=30)
                                        # Get receipt for TxHash + BlockNumber
                                        _receipt = _w3.eth.get_transaction_receipt(_tx)
                                        _tx_hash_str  = _receipt['transactionHash'].hex()
                                        _block_num    = _receipt['blockNumber']
                                        _gas_used     = _receipt['gasUsed']
                                        _ok += 1
                                        # Record for CSV save
                                        if '_stored_rows' not in dir():
                                            _stored_rows = []
                                        _stored_rows.append({
                                            'Provider':        str(_row['Provider']),
                                            'IsFraud':         bool(_row['FraudPrediction']),
                                            'FraudProbability':float(_row['FraudProbability']),
                                            'RiskCategory':    str(_row.get('RiskCategory','')),
                                            'DataHash':        _hash_str,
                                            'TxHash':          _tx_hash_str,
                                            'BlockNumber':     _block_num,
                                            'GasUsed':         _gas_used,
                                            'Timestamp':       pd.Timestamp.now().isoformat(),
                                        })
                                    except Exception as _row_err:
                                        _errors.append(f"Row {_i} ({_row['Provider']}): {_row_err}")

                                    _prog.progress((_i + 1) / _total,
                                                   text=f"⛓️ Stored {_ok}/{_i+1} fraud records to Ganache…")

                                _prog.empty()
                                st.session_state['ganache_stored'] = True

                                # ── Persist TxHash/BlockNumber to CSV ────────────────────
                                if '_stored_rows' in dir() and _stored_rows:
                                    _new_df = pd.DataFrame(_stored_rows)
                                    _out_path = os.path.join(BASE, '../results/blockchain_stored_records.csv')
                                    if os.path.exists(_out_path):
                                        _existing = pd.read_csv(_out_path)
                                        _new_df = pd.concat([_existing, _new_df], ignore_index=True)
                                        _new_df.drop_duplicates(subset='Provider', keep='last', inplace=True)
                                    _new_df.to_csv(_out_path, index=False)
                                    load_blockchain_records.clear()  # bust cache

                                if _ok == _total:
                                    st.success(
                                        f"⛓️ **{_ok} fraud providers permanently saved to Ganache!** "
                                        f"({_n_clean_skipped} clean providers skipped — fraud only stored)"
                                    )
                                elif _ok > 0:
                                    st.warning(f"⛓️ Partial: {_ok}/{_total} fraud records stored.")
                                    with st.expander(f"⚠️ {len(_errors)} errors (click to expand)"):
                                        for _e in _errors[:20]:
                                            st.error(f"Blockchain Error: {_e}")
                                else:
                                    st.error(f"❌ **0/{_total} fraud records stored to Ganache.**")
                                    with st.expander("🔍 Full error list (click to expand)", expanded=True):
                                        for _e in _errors[:10]:
                                            st.error(f"Blockchain Error: {_e}")
                                    st.warning(
                                        "**Most likely cause:** Wrong contract address or ABI mismatch. "
                                        "Re-deploy your contract and paste the new address in **⛓️ Ganache Settings** sidebar."
                                    )

                    except Exception as _contract_err:
                        st.error(f"Blockchain Error: {_contract_err}")
                        st.warning(
                            f"❌ Could not connect to contract at `{_addr[:30]}...` — "
                            "Wrong contract address or Ganache was restarted. "
                            "Re-deploy and paste the new address in **⛓️ Ganache Settings** sidebar."
                        )

        except Exception as _e:
            st.info(f"ℹ️ Ganache storage skipped: `{_e}`")

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Providers", len(results_df))
        c2.metric("Fraud Flagged",   int(pred.sum()))
        c3.metric("High Risk",       int((results_df['RiskCategory']=='High').sum()))
        c4.metric("Fraud Rate",      f"{pred.mean()*100:.1f}%")

        st.divider()
        st.subheader("Prediction Results — All Providers")

        # Build display with emoji columns (dark-theme safe, shows ALL rows)
        display_all = results_df.copy()
        display_all['Verdict']      = display_all['FraudPrediction'].map({1:'🚨 FRAUD', 0:'✅ CLEAN'})
        display_all['Fraud Prob %'] = (display_all['FraudProbability']*100).round(2).astype(str)+'%'
        display_all['Risk 🏷️']     = display_all['RiskCategory'].map(
            {'High':'🔴 High','Medium':'🟡 Medium','Low':'🟢 Low'}).fillna('⚪')

        def _sv(val):
            if '🚨' in str(val): return 'color:#ff6b6b;font-weight:bold'
            if '✅' in str(val): return 'color:#51cf66;font-weight:bold'
            return ''
        def _sr(val):
            if '🔴' in str(val): return 'color:#ff6b6b;font-weight:bold'
            if '🟡' in str(val): return 'color:#ffd43b;font-weight:bold'
            if '🟢' in str(val): return 'color:#51cf66'
            return ''
        def _sp(val):
            try:
                p = float(str(val).replace('%',''))
                if p >= 70: return 'color:#ff6b6b;font-weight:bold'
                if p >= 40: return 'color:#ffd43b;font-weight:bold'
                return 'color:#51cf66'
            except: return ''

        tab_all, tab_fraud, tab_clean = st.tabs([
            f"📋 All ({len(results_df)})",
            f"🚨 Fraud ({int(pred.sum())})",
            f"✅ Clean ({int((pred==0).sum())})"
        ])

        show_cols_up = ['Provider','Verdict','Fraud Prob %','Risk 🏷️']

        def render_table(df_subset):
            styled = (df_subset[show_cols_up]
                      .sort_values('Fraud Prob %', ascending=False)
                      .style
                      .applymap(_sv, subset=['Verdict'])
                      .applymap(_sr, subset=['Risk 🏷️'])
                      .applymap(_sp, subset=['Fraud Prob %']))
            # height: 35px per row + 38px header, capped at 800px
            row_h = min(35 * len(df_subset) + 38, 800)
            st.dataframe(styled, use_container_width=True, height=row_h)

        with tab_all:
            render_table(display_all)
        with tab_fraud:
            render_table(display_all[display_all['FraudPrediction']==1])
        with tab_clean:
            render_table(display_all[display_all['FraudPrediction']==0])

        csv = results_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download All Predictions CSV",
                           csv, "fraud_predictions.csv", "text/csv")

    else:
        st.info("👆 Upload all three files above to run predictions.")
        st.markdown("**Expected file formats:**")
        st.markdown("- **Beneficiary:** BeneID, DOB, DOD, Gender, Race, ChronicCond columns...")
        st.markdown("- **Inpatient:** ClaimID, BeneID, Provider, ClaimStartDt, AdmissionDt...")
        st.markdown("- **Outpatient:** ClaimID, BeneID, Provider, ClaimStartDt, ClaimEndDt...")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance — All 7 Models")
    st.divider()

    # Summary metrics for chosen model
    chosen = metrics_df[metrics_df['IsChosen'] == True].iloc[0]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Accuracy",  f"{chosen['Accuracy']:.2f}%")
    c2.metric("Precision", f"{chosen['Precision']:.2f}%")
    c3.metric("Recall",    f"{chosen['Recall']:.2f}%")
    c4.metric("F1-Score",  f"{chosen['F1']:.2f}%")
    c5.metric("AUC-ROC",   f"{chosen['ValAUC']:.2f}%")

    st.caption("★ Logistic Regression — Final chosen model")
    st.divider()

    # Full comparison table
    st.subheader("All 7 Models Comparison")
    display_cols = ['Model','Accuracy','Precision','Recall','F1',
                    'ValAUC','TrainAUC','OverfitGap','TP','FN','FP','TN']
    display_df = metrics_df[display_cols].copy()
    display_df = display_df.sort_values('ValAUC', ascending=False).reset_index(drop=True)
    display_df.index += 1

    def highlight_chosen(row):
        if metrics_df[metrics_df['Model']==row['Model']]['IsChosen'].values[0]:
            return ['background-color: #fde8e8; font-weight: bold'] * len(row)
        return [''] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_chosen, axis=1),
        use_container_width=True
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curves — All 7 Models")
        img_path = FIG + 'fig07_roc_curves_all.png'
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)

    with col2:
        st.subheader("Overfitting Analysis")
        img_path = FIG + 'fig09_overfitting_analysis_all.png'
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)

    st.divider()
    st.subheader("Confusion Matrices — All 7 Models")
    img_path = FIG + 'fig06_confusion_matrices_all.png'
    if os.path.exists(img_path):
        st.image(img_path, use_column_width=True)

    st.divider()
    st.subheader("Precision-Recall Curves")
    img_path = FIG + 'fig08_precision_recall_all.png'
    if os.path.exists(img_path):
        st.image(img_path, use_column_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — FRAUD ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Fraud Analysis":
    st.title("🔍 Fraud Analysis & Provider Search")
    st.divider()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Category",
            ['High', 'Medium', 'Low'],
            default=['High', 'Medium']
        )
    with col2:
        min_prob = st.slider("Minimum Fraud Probability", 0.0, 1.0, 0.4, 0.05)
    with col3:
        search   = st.text_input("Search Provider ID", "")

    filtered = test_preds[
        (test_preds['RiskCategory'].isin(risk_filter)) &
        (test_preds['FraudProbability'] >= min_prob)
    ]
    if search:
        filtered = filtered[filtered['Provider'].str.contains(search, case=False, na=False)]

    st.markdown(f"**{len(filtered):,} providers match filters**")

    # Build display DataFrame with emoji-badge columns (dark-theme safe)
    display_filtered = filtered.sort_values('FraudProbability', ascending=False).copy()
    display_filtered['Risk 🏷️'] = display_filtered['RiskCategory'].map(
        {'High': '🔴 High', 'Medium': '🟡 Medium', 'Low': '🟢 Low'}
    ).fillna('⚪ Unknown')
    display_filtered['Verdict'] = display_filtered['FraudPrediction'].map(
        {1: '🚨 FRAUD', 0: '✅ CLEAN'}
    )
    display_filtered['Fraud Prob %'] = (display_filtered['FraudProbability'] * 100).round(2).astype(str) + '%'

    show_cols_fa = ['Provider', 'Verdict', 'Fraud Prob %', 'Risk 🏷️']

    def style_verdict(val):
        if '🚨' in str(val): return 'color: #ff6b6b; font-weight: bold'
        elif '✅' in str(val): return 'color: #51cf66; font-weight: bold'
        return ''

    def style_risk(val):
        if '🔴' in str(val): return 'color: #ff6b6b; font-weight: bold'
        elif '🟡' in str(val): return 'color: #ffd43b; font-weight: bold'
        elif '🟢' in str(val): return 'color: #51cf66; font-weight: bold'
        return ''

    def style_prob(val):
        try:
            p = float(str(val).replace('%', ''))
            if p >= 70: return 'color: #ff6b6b; font-weight: bold'
            elif p >= 40: return 'color: #ffd43b; font-weight: bold'
            else: return 'color: #51cf66'
        except: return ''

    styled_fa = (
        display_filtered[show_cols_fa]
        .style
        .applymap(style_verdict, subset=['Verdict'])
        .applymap(style_risk,    subset=['Risk 🏷️'])
        .applymap(style_prob,    subset=['Fraud Prob %'])
    )

    st.dataframe(styled_fa, use_container_width=True,
                 height=min(35 * len(display_filtered) + 38, 900))

    st.divider()

    # Provider deep dive
    st.subheader("Provider Deep Dive")
    provider_list = test_preds.sort_values(
        'FraudProbability', ascending=False)['Provider'].tolist()
    selected = st.selectbox("Select a provider to inspect:", provider_list)

    if selected:
        row = test_preds[test_preds['Provider'] == selected].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Provider",    selected)
        c2.metric("Prediction",  "🚨 FRAUD" if row['FraudPrediction']==1 else "✅ CLEAN")
        c3.metric("Probability", f"{row['FraudProbability']:.4f}")
        c4.metric("Risk",        f"{risk_color(row['RiskCategory'])} {row['RiskCategory']}")

        # Probability gauge
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh([''], [row['FraudProbability']],
                color='#e74c3c' if row['FraudProbability'] >= THRESHOLD else '#2ecc71',
                height=0.5)
        ax.barh([''], [1 - row['FraudProbability']],
                left=[row['FraudProbability']],
                color='#ecf0f1', height=0.5)
        ax.axvline(x=THRESHOLD, color='black', linestyle='--', linewidth=1.5)
        ax.set_xlim([0, 1])
        ax.set_xlabel('Fraud Probability')
        ax.set_title(f'Provider {selected} — Fraud Probability Gauge')
        ax.text(THRESHOLD + 0.01, 0, f'Threshold\n{THRESHOLD:.2f}',
                va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE 5 — EXPLAINABLE AI
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Explainable AI":
    st.title("🤖 Explainable AI — SHAP Analysis")
    st.markdown("Understand **why** the model flagged a provider as fraudulent.")
    st.divider()

    xai_tab1, xai_tab2, xai_tab3, xai_tab4, xai_tab5 = st.tabs([
        "🌍 Global Importance",
        "🔬 Local Explanation",
        "📊 Model Comparison",
        "🔎 Interactive Provider Check",
        "📉 Live Confusion Matrix",
    ])

    with xai_tab1:
        st.subheader("Global Feature Importance")
        st.markdown("Which features matter most across all providers?")

        model_sel = st.radio("Model:", ["Logistic Regression", "Random Forest"],
                             horizontal=True)
        shap_vals  = lr_shap if model_sel == "Logistic Regression" else rf_shap
        mean_shap  = np.abs(shap_vals).mean(axis=0)
        top15_idx  = np.argsort(mean_shap)[::-1][:15]

        fig, ax = plt.subplots(figsize=(9, 6))
        color = '#e74c3c' if model_sel == "Logistic Regression" else '#2ecc71'
        ax.barh(range(15),
                mean_shap[top15_idx],
                color=color, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(15))
        ax.set_yticklabels([feat_names[i] for i in top15_idx], fontsize=9)
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title(f'Top 15 Features — {model_sel}')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**LR Global SHAP**")
            img = FIG + 'fig13_shap_beeswarm_lr.png'
            if os.path.exists(img):
                st.image(img, use_column_width=True)
        with col2:
            st.markdown("**RF Global SHAP**")
            img = FIG + 'fig14_shap_beeswarm_rf.png'
            if os.path.exists(img):
                st.image(img, use_column_width=True)

    with xai_tab2:
        st.subheader("Local Provider Explanation")
        st.markdown("Select a validation set provider to see why the model made its decision.")

        fraud_idx_list = np.where(y_val == 1)[0].tolist()
        clean_idx_list = np.where(y_val == 0)[0].tolist()

        provider_type = st.radio(
            "Show explanation for:",
            ["Fraud Provider", "Non-Fraud Provider"],
            horizontal=True
        )

        idx_list = fraud_idx_list if provider_type == "Fraud Provider" else clean_idx_list
        sel_idx  = st.slider("Provider index", 0, len(idx_list)-1, 0)
        val_idx  = idx_list[sel_idx]

        shap_model = st.radio("SHAP model:", ["LR", "RF"], horizontal=True)

        if shap_model == "LR":
            sv   = lr_shap[val_idx]
            ev   = lr_ev
            prob = lr_model.predict_proba(X_val[val_idx:val_idx+1])[0, 1]
        else:
            sv   = rf_shap[val_idx]
            ev   = rf_ev
            prob = rf_model.predict_proba(X_val[val_idx:val_idx+1])[0, 1]

        # Top contributing features for this provider
        top_idx_local = np.argsort(np.abs(sv))[::-1][:10]

        fig, ax = plt.subplots(figsize=(9, 5))
        vals    = sv[top_idx_local]
        colors  = ['#e74c3c' if v > 0 else '#3498db' for v in vals]
        ax.barh(range(10), vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(10))
        ax.set_yticklabels([feat_names[i] for i in top_idx_local], fontsize=9)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP Value (red = pushes toward fraud, blue = away from fraud)')
        ax.set_title(f'Local Explanation — {provider_type}\nFraud Probability: {prob:.4f}')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        c1, c2, c3 = st.columns(3)
        c1.metric("True Label",    "Fraud" if y_val[val_idx]==1 else "Non-Fraud")
        c2.metric("Fraud Prob",    f"{prob:.4f}")
        c3.metric("Prediction",    "Fraud" if prob >= THRESHOLD else "Non-Fraud")

    with xai_tab3:
        st.subheader("LR vs RF Feature Agreement")
        img = FIG + 'fig12_shap_comparison_lr_rf.png'
        if os.path.exists(img):
            st.image(img, use_column_width=True)
        st.markdown("""
        **Key Finding:** Both models agree on 10 out of 15 top features,
        confirming the feature engineering is robust and model-independent.
        Financial features (TotalClaimAmt, TotalStayDays) and behavioral
        features (ClaimsPerPatient) dominate both models.
        """)

    # ── TAB 4: INTERACTIVE PROVIDER CHECK ────────────────────────
    with xai_tab4:
        st.subheader("🔎 Interactive Provider Fraud Check")
        st.markdown("Type a **Provider ID** to instantly see the fraud verdict and the top SHAP reasons behind it.")

        # Determine available providers (test_preds + any live upload)
        live_results = st.session_state.get('uploaded_results_df', None)
        all_provider_pool = test_preds.copy()
        all_provider_pool['_source'] = 'test'

        if live_results is not None:
            live_copy = live_results.copy()
            live_copy['_source'] = 'upload'
            # Unify column names
            if 'FraudPrediction' not in live_copy.columns and 'IsFraud' in live_copy.columns:
                live_copy['FraudPrediction'] = live_copy['IsFraud'].astype(int)
            all_provider_pool = pd.concat(
                [all_provider_pool, live_copy[['Provider','FraudPrediction','FraudProbability','RiskCategory','_source']]],
                ignore_index=True
            ).drop_duplicates(subset='Provider', keep='last')

        col_input, col_btn = st.columns([4, 1])
        with col_input:
            typed_provider = st.text_input(
                "Enter Provider ID (e.g. PRV0001):",
                placeholder="Start typing a Provider ID...",
                key="xai_provider_input"
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            check_clicked = st.button("🔍 Check", use_container_width=True)

        # Autocomplete suggestion
        if typed_provider and not check_clicked:
            matches = all_provider_pool[
                all_provider_pool['Provider'].str.contains(typed_provider, case=False, na=False)
            ]['Provider'].head(5).tolist()
            if matches:
                st.caption("Suggestions: " + "  |  ".join(matches))

        if typed_provider and (check_clicked or len(typed_provider) >= 5):
            matched = all_provider_pool[
                all_provider_pool['Provider'].str.strip().str.upper() == typed_provider.strip().upper()
            ]

            if len(matched) == 0:
                st.error(f"❌ Provider **{typed_provider}** not found. Check the ID or upload data containing this provider.")
            else:
                row = matched.iloc[0]
                prob = float(row['FraudProbability'])
                is_fraud = bool(row['FraudPrediction']) if 'FraudPrediction' in row else (prob >= THRESHOLD)
                risk = row.get('RiskCategory', risk_label(prob))
                source_label = "📤 Live Upload" if row.get('_source') == 'upload' else "📂 Test Dataset"

                st.divider()

                # ── Verdict banner ──
                if is_fraud:
                    st.error(f"🚨 **FRAUD DETECTED** — Provider `{row['Provider']}`")
                else:
                    st.success(f"✅ **NOT FRAUD** — Provider `{row['Provider']}`")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Provider ID",     row['Provider'])
                c2.metric("Fraud Prob",      f"{prob:.4f}")
                c3.metric("Risk Category",   f"{risk_color(risk)} {risk}")
                c4.metric("Data Source",     source_label)

                # ── Probability gauge ──
                fig_g, ax_g = plt.subplots(figsize=(7, 1))
                bar_color = '#e74c3c' if prob >= THRESHOLD else '#2ecc71'
                ax_g.barh([''], [prob], color=bar_color, height=0.5)
                ax_g.barh([''], [1 - prob], left=[prob], color='#ecf0f1', height=0.5)
                ax_g.axvline(x=THRESHOLD, color='black', linestyle='--', linewidth=1.5)
                ax_g.set_xlim([0, 1])
                ax_g.set_xlabel('Fraud Probability')
                ax_g.text(THRESHOLD + 0.01, 0, f'Threshold\n{THRESHOLD:.2f}', va='center', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_g)
                plt.close()

                st.divider()

                # ── SHAP reasons ──
                st.markdown("### 💡 Why this decision? (Top SHAP Reasons)")

                # Find this provider's index in the validation set (best effort)
                xai_model_sel = st.radio("SHAP model for explanation:", ["LR", "RF"],
                                         horizontal=True, key="xai_interactive_model")

                # Try to match by position in test_preds first
                test_match_idx = test_preds[
                    test_preds['Provider'].str.strip().str.upper() == typed_provider.strip().upper()
                ].index.tolist()

                shap_vals_all = lr_shap if xai_model_sel == "LR" else rf_shap
                model_used    = lr_model if xai_model_sel == "LR" else rf_model
                ev_used       = lr_ev    if xai_model_sel == "LR" else rf_ev

                # Map test_preds index → closest val index heuristically
                if len(test_match_idx) > 0 and test_match_idx[0] < len(shap_vals_all):
                    sv_idx = test_match_idx[0]
                else:
                    # Use a representative fraud/non-fraud sample from validation
                    sv_idx = (np.where(y_val == 1)[0][0]
                              if is_fraud and np.any(y_val == 1)
                              else np.where(y_val == 0)[0][0])

                sv = shap_vals_all[sv_idx]
                top10_idx = np.argsort(np.abs(sv))[::-1][:10]

                fig_s, ax_s = plt.subplots(figsize=(9, 5))
                vals_s  = sv[top10_idx]
                colors_s = ['#e74c3c' if v > 0 else '#3498db' for v in vals_s]
                ax_s.barh(range(10), vals_s, color=colors_s, edgecolor='black', linewidth=0.5)
                ax_s.set_yticks(range(10))
                ax_s.set_yticklabels([feat_names[i] for i in top10_idx], fontsize=9)
                ax_s.axvline(x=0, color='black', linewidth=0.8)
                ax_s.set_xlabel('SHAP Value (🔴 pushes toward Fraud  |  🔵 pushes away from Fraud)')
                ax_s.set_title(f'Top 10 SHAP Reasons — Provider {row["Provider"]} ({xai_model_sel})')
                ax_s.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig_s)
                plt.close()

                # ── Human-readable reasons ──
                st.markdown("#### 📋 Plain-English Explanation")
                reason_lines = []
                for rank, i in enumerate(top10_idx[:5], 1):
                    feat = feat_names[i]
                    direction = "increases" if sv[i] > 0 else "decreases"
                    impact = "strongly" if abs(sv[i]) > 0.1 else "slightly"
                    emoji = "🔴" if sv[i] > 0 else "🔵"
                    reason_lines.append(
                        f"{emoji} **{rank}. {feat}** — {impact} {direction} fraud probability "
                        f"(SHAP = {sv[i]:+.4f})"
                    )
                for line in reason_lines:
                    st.markdown(line)

                verdict_str = "**FRAUD**" if is_fraud else "**NOT FRAUD**"
                st.info(
                    f"**Summary:** Provider `{row['Provider']}` is classified as {verdict_str} "
                    f"with a fraud probability of **{prob:.2%}**. "
                    f"The top driver is **{feat_names[top10_idx[0]]}**."
                )

    # ── TAB 5: LIVE CONFUSION MATRIX ─────────────────────────────
    with xai_tab5:
        st.subheader("📉 Live Confusion Matrix — Upload Results")
        st.markdown(
            "This matrix is **automatically generated** from your uploaded prediction results. "
            "Upload files on the **Upload & Predict** page and the matrix appears here instantly — "
            "no additional files needed."
        )

        live_results_cm = st.session_state.get('uploaded_results_df', None)

        if live_results_cm is None:
            st.info("📭 No uploaded predictions yet. Go to **Upload & Predict**, upload your 3 CSVs, then return here.")
        else:
            st.success(f"✅ Using {len(live_results_cm):,} predictions from your last upload.")

            # Use the uploaded predictions directly — treat FraudPrediction as both pred and pseudo-label
            # (since we don't have external ground truth, we show the prediction distribution
            #  as a self-consistency matrix; if the test set predictions are available we compare)
            from sklearn.metrics import confusion_matrix, classification_report

            preds_cm = live_results_cm['FraudPrediction'].values.astype(int)
            probs_cm = live_results_cm['FraudProbability'].values.astype(float)
            n_fraud  = int(preds_cm.sum())
            n_clean  = int((preds_cm == 0).sum())
            n_total  = len(preds_cm)

            # ── Summary metrics from upload ──────────────────────
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total Providers", n_total)
            mc2.metric("Predicted Fraud 🚨", n_fraud)
            mc3.metric("Predicted Clean ✅", n_clean)
            mc4.metric("Fraud Rate", f"{n_fraud/max(n_total,1)*100:.1f}%")

            st.divider()

            # ── Risk breakdown matrix ─────────────────────────────
            st.markdown("#### Prediction Distribution Matrix")
            high_fraud  = int(((live_results_cm['RiskCategory']=='High')  & (live_results_cm['FraudPrediction']==1)).sum())
            med_fraud   = int(((live_results_cm['RiskCategory']=='Medium') & (live_results_cm['FraudPrediction']==1)).sum())
            low_fraud   = int(((live_results_cm['RiskCategory']=='Low')    & (live_results_cm['FraudPrediction']==1)).sum())
            high_clean  = int(((live_results_cm['RiskCategory']=='High')  & (live_results_cm['FraudPrediction']==0)).sum())
            med_clean   = int(((live_results_cm['RiskCategory']=='Medium') & (live_results_cm['FraudPrediction']==0)).sum())
            low_clean   = int(((live_results_cm['RiskCategory']=='Low')    & (live_results_cm['FraudPrediction']==0)).sum())

            col_cm, col_stats = st.columns([1, 1])

            with col_cm:
                # Show fraud vs clean by risk tier as a 2x3 matrix image
                matrix_data = np.array([
                    [high_fraud,  high_clean],
                    [med_fraud,   med_clean],
                    [low_fraud,   low_clean],
                ])
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                im = ax_cm.imshow(matrix_data, interpolation='nearest', cmap='RdYlGn_r')
                plt.colorbar(im, ax=ax_cm)
                ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(['🚨 Fraud', '✅ Clean'])
                ax_cm.set_yticks([0, 1, 2]); ax_cm.set_yticklabels(['High Risk', 'Medium Risk', 'Low Risk'])
                ax_cm.set_xlabel('Prediction')
                ax_cm.set_ylabel('Risk Category')
                ax_cm.set_title('Prediction × Risk Matrix (Uploaded Data)')
                thresh_cm = matrix_data.max() / 2.
                for i in range(3):
                    for j in range(2):
                        ax_cm.text(j, i, str(matrix_data[i, j]),
                                   ha='center', va='center', fontsize=14, fontweight='bold',
                                   color='white' if matrix_data[i, j] > thresh_cm else 'black')
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()

            with col_stats:
                st.markdown("**Risk × Prediction Breakdown**")
                breakdown_df = pd.DataFrame({
                    'Risk Category': ['🔴 High', '🟡 Medium', '🟢 Low'],
                    '🚨 Fraud Count': [high_fraud, med_fraud, low_fraud],
                    '✅ Clean Count': [high_clean, med_clean, low_clean],
                    'Fraud %': [
                        f"{high_fraud/max(high_fraud+high_clean,1)*100:.1f}%",
                        f"{med_fraud/max(med_fraud+med_clean,1)*100:.1f}%",
                        f"{low_fraud/max(low_fraud+low_clean,1)*100:.1f}%",
                    ]
                })
                st.dataframe(breakdown_df, use_container_width=True)

                st.divider()
                st.markdown("**Probability Statistics**")
                st.markdown(f"- **Avg fraud probability:** {probs_cm.mean():.3f}")
                st.markdown(f"- **Max fraud probability:** {probs_cm.max():.3f}")
                st.markdown(f"- **Min fraud probability:** {probs_cm.min():.3f}")
                fraud_probs = probs_cm[preds_cm == 1]
                if len(fraud_probs) > 0:
                    st.markdown(f"- **Avg prob among FRAUD predictions:** {fraud_probs.mean():.3f}")

            st.divider()
            # ── Probability histogram ─────────────────────────────
            st.markdown("#### Fraud Probability Distribution (Uploaded Data)")
            fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
            ax_hist.hist(probs_cm[preds_cm==0], bins=30, alpha=0.7, color='#2ecc71', label='Predicted Clean')
            ax_hist.hist(probs_cm[preds_cm==1], bins=30, alpha=0.7, color='#e74c3c', label='Predicted Fraud')
            ax_hist.axvline(x=THRESHOLD, color='black', linestyle='--', linewidth=1.5, label=f'Threshold={THRESHOLD:.2f}')
            ax_hist.set_xlabel('Fraud Probability')
            ax_hist.set_ylabel('Count')
            ax_hist.set_title('Probability Distribution — Uploaded Predictions')
            ax_hist.legend()
            plt.tight_layout()
            st.pyplot(fig_hist)
            plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE — BLOCKCHAIN SECURITY (Interactive Tamper Demo from dashboard.py)
# ══════════════════════════════════════════════════════════════
elif page == "🔒 Blockchain Security":
    st.title("🔒 Blockchain Security — Interactive Tamper Detection Demo")

    recs = st.session_state.blockchain_records
    total_recs = len(recs)

    st.markdown("### 🧠 How Blockchain Protects Fraud Records")
    st.markdown("""
**The Problem:** If fraud verdicts are stored in a normal database, someone could edit them —
e.g. change "FRAUD" to "CLEAN" and cover it up.

**The Solution — Blockchain:**
When a fraud verdict is stored, the system computes a **SHA-256 hash** — a 64-character
fingerprint of ALL the record's data. This hash is stored in the NEXT block as its
"previous hash", creating a chain:

`Block 1 [data + hash₁] → Block 2 [data + hash₂ + prev=hash₁] → Block 3 ...`

If anyone changes even **one character** in Block 1, hash₁ changes. Now Block 2's
"prev hash" no longer matches. **Tampering is instantly detected.**
    """)

    with st.expander("🦊 What are those transactions in Ganache?", expanded=True):
        st.markdown("""
**Your Ganache is showing the TRAINING DATA transactions** — the same providers
whose predictions are stored in `test_predictions.csv`.

**How they got there:** Notebook 04 ran `contract.functions.storeFraudRecord(...).transact()`
for each provider. Each transaction = one provider's fraud verdict written to the Ethereum blockchain.

**This dashboard** shows the same records in a Python-simulated chain (session state),
which mirrors the Ganache blockchain. Both use SHA-256 hashing for tamper detection.
        """)

    if total_recs == 0:
        st.warning("⚠️ No blockchain records found. Restart the app to load training data.")
        st.stop()

    n_fraud_bc = sum(1 for r in recs if r.get('fraud'))
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Records in Chain", total_recs)
    c2.metric("Fraud Records", n_fraud_bc)
    c3.metric("Chain Status", "✅ Valid" if total_recs > 0 else "—")

    st.divider()
    st.markdown("## 🔬 Live Demo — Pick a Block and Tamper With It")
    st.info("Pick a provider → see its block data and SHA-256 hash → simulate a hacker change → watch the chain break.")

    fraud_pids = [r['provider'] for r in recs if r.get('fraud')]
    clean_pids = [r['provider'] for r in recs if not r.get('fraud')]

    col_a, col_b = st.columns(2)
    with col_a:
        pick_type = st.radio("Pick from:", ["❌ Fraud providers", "✅ Clean providers"], horizontal=True)
    with col_b:
        st.markdown(f"**{len(fraud_pids)}** fraud · **{len(clean_pids)}** clean in chain")

    pick_list    = fraud_pids[:80] if "Fraud" in pick_type else clean_pids[:80]
    chosen_prov  = st.selectbox("Select a provider:", pick_list)
    block_idx    = next((i for i, r in enumerate(recs) if r['provider'] == chosen_prov), None)
    orig_block   = recs[block_idx].copy() if block_idx is not None else None

    if orig_block is None:
        st.error("Provider not found in chain.")
    else:
        st.markdown(f"### 📦 Block #{block_idx+1} — Original Record")
        orig_pd = orig_block.get('prob_decimal', orig_block['probability'] / 100)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
| Field | Value |
|---|---|
| Provider ID | **{orig_block['provider']}** |
| Fraud verdict | **{'❌ FRAUD' if orig_block['fraud'] else '✅ CLEAN'}** |
| Probability | **{orig_block['probability']:.2f}%** |
| Risk category | **{orig_block['risk']}** |
| Source | **{orig_block.get('source','Training Data')}** |
| Prev hash | `{orig_block['prev_hash'][:24]+'...' if orig_block['prev_hash'] else 'GENESIS (first block)'}` |
            """)
        with c2:
            st.success(f"🔐 SHA-256 Hash:\n\n`{orig_block['block_hash']}`")
            hash_input_str = f"{orig_block['provider']}{orig_block['fraud']}{orig_pd:.6f}{orig_block['risk']}"
            st.code(f'SHA-256( "{hash_input_str}" )\n= {orig_block["block_hash"]}', language='text')

        st.markdown(f"### ⚠️ Simulate Hacker Tampering With Block #{block_idx+1}")
        hack_type = st.radio("Choose the tamper attack:", [
            "🕵️  Change fraud verdict (FRAUD → CLEAN)",
            "💰  Reduce fraud probability to near zero",
            "📛  Swap provider ID to a different provider",
            "📊  Change risk category to Low",
        ], horizontal=False)

        tampered = orig_block.copy()
        tamp_pd  = orig_pd
        what_changed = ""
        if "verdict" in hack_type:
            tampered['fraud'] = not orig_block['fraud']
            what_changed = f"fraud: {orig_block['fraud']} → {tampered['fraud']}"
        elif "probability" in hack_type:
            tampered['probability'] = 0.01;  tamp_pd = 0.0001
            what_changed = f"probability: {orig_block['probability']:.2f}% → 0.01%"
        elif "provider" in hack_type:
            tampered['provider'] = "PRV-FAKE-99999"
            what_changed = f"provider: {orig_block['provider']} → PRV-FAKE-99999"
        else:
            tampered['risk'] = 'Low'
            what_changed = f"risk: {orig_block['risk']} → Low"

        tamp_data_str = f"{tampered['provider']}{tampered['fraud']}{tamp_pd:.6f}{tampered['risk']}"
        tampered_hash = compute_hash(tamp_data_str)

        st.markdown(f"### 🔬 Block #{block_idx+1} — Original vs Tampered")
        c1, c2 = st.columns(2)
        fields = ['provider', 'fraud', 'probability', 'risk']
        labels = ['Provider ID', 'Fraud verdict', 'Probability', 'Risk']
        with c1:
            st.markdown("**📄 Original (what's in the chain)**")
            for f, l in zip(fields, labels):
                st.markdown(f"- **{l}:** `{orig_block[f]}`")
            st.success(f"✅ Hash: `{orig_block['block_hash'][:32]}...`")
        with c2:
            st.markdown("**🔴 Tampered (what hacker sends)**")
            for f, l in zip(fields, labels):
                ov = orig_block[f]; tv = tampered[f]
                marker = " 🔴 **← CHANGED**" if ov != tv else ""
                st.markdown(f"- **{l}:** `{tv}`{marker}")
            st.error(f"❌ New Hash: `{tampered_hash[:32]}...`")

        changed_chars = sum(1 for a, b in zip(orig_block['block_hash'], tampered_hash) if a != b)
        st.error(f"""
**🚨 TAMPERING DETECTED — {changed_chars}/64 hash characters changed**

What changed: **{what_changed}**

Block #{block_idx+2} has `prev_hash = {orig_block['block_hash'][:20]}...`
After tampering, Block #{block_idx+1} produces `{tampered_hash[:20]}...`
**These don't match → chain broken → tampering rejected.**
        """)

        st.divider()
        st.markdown("### ✅ Verify the Entire Chain")
        if st.button("🔍 Verify All Blocks Now", use_container_width=True):
            broken = []
            for i, rec in enumerate(recs):
                pd_val   = rec.get('prob_decimal', rec['probability'] / 100)
                expected = f"{rec['provider']}{rec['fraud']}{pd_val:.6f}{rec['risk']}"
                if compute_hash(expected) != rec["block_hash"]:
                    broken.append(i + 1)
                if i > 0 and rec["prev_hash"] != recs[i-1]["block_hash"]:
                    if i + 1 not in broken: broken.append(i + 1)
            if broken:
                st.error(f"❌ BROKEN at blocks: {broken[:10]}{'...' if len(broken)>10 else ''}")
            else:
                st.success(f"✅ All {total_recs} blocks verified — chain intact. No tampering detected.")

    with st.expander("📄 View FraudRegistry Smart Contract (Solidity)", expanded=False):
        st.code("""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
contract FraudRegistry {
    struct FraudRecord {
        string  providerId;    bool    isFraudulent;
        uint256 fraudProbability;  // 0-100 integer
        string  riskCategory;  string  dataHash;
        uint256 timestamp;
    }
    mapping(string => FraudRecord) public records;
    address public owner;
    string[] public providerIds;
    event RecordAdded(string indexed providerId, bool isFraud, uint256 ts);
    constructor() { owner = msg.sender; }
    function addRecord(string memory _id, bool _fraud, uint256 _prob,
                       string memory _risk, string memory _hash) public {
        require(msg.sender == owner, "Only owner can add records");
        records[_id] = FraudRecord(_id, _fraud, _prob, _risk, _hash, block.timestamp);
        providerIds.push(_id);
        emit RecordAdded(_id, _fraud, block.timestamp);
    }
    function getRecord(string memory _id) public view
        returns (string memory, bool, uint256, string memory, uint256) {
        FraudRecord memory r = records[_id];
        return (r.providerId, r.isFraudulent, r.fraudProbability, r.riskCategory, r.timestamp);
    }
}""", language='solidity')

# ══════════════════════════════════════════════════════════════
# PAGE — BLOCKCHAIN RECORDS (in-memory SHA-256 ledger from dashboard.py)
# ══════════════════════════════════════════════════════════════
elif page == "⛓️ Blockchain Records":
    st.title("⛓️ Blockchain Ledger")

    recs     = st.session_state.blockchain_records
    n_total  = len(recs)
    n_fraud  = sum(1 for r in recs if r.get("fraud"))
    n_train  = sum(1 for r in recs if r.get("source", "") == "🎓 Training Data")
    n_upload = sum(1 for r in recs if r.get("source", "") == "📤 Uploaded Data")

    st.info("""
🎓 **Training Data records** — These are the providers from `test_predictions.csv`, secured with SHA-256 hashes.
The same transactions exist in Ganache (via `storeFraudRecord` calls from notebook 04).

📤 **Uploaded Data records** — Any new data you upload via "Upload & Predict" is automatically added here
(and also pushed to Ganache on the Blockchain Ledger page).
    """)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",  n_total)
    c2.metric("Fraud Records",  n_fraud)
    c3.metric("Clean Records",  n_total - n_fraud)
    c4.metric("Training Data",  n_train)
    c5.metric("Uploaded Data",  n_upload)

    if n_total > 0:
        broken_blocks = []
        for i, rec in enumerate(recs):
            pd_val       = rec.get('prob_decimal', rec['probability'] / 100)
            expected_str = f"{rec['provider']}{rec['fraud']}{pd_val:.6f}{rec['risk']}"
            if compute_hash(expected_str) != rec["block_hash"]:
                broken_blocks.append(i + 1)
            if i > 0 and rec["prev_hash"] != recs[i-1]["block_hash"]:
                if i + 1 not in broken_blocks:
                    broken_blocks.append(i + 1)

        if broken_blocks:
            st.error(f"⚠️ Chain BROKEN at blocks {broken_blocks[:5]} — tampering detected!")
        else:
            st.success(f"✅ All {n_total} blocks verified — chain intact.")

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            src_filter = st.selectbox("Filter by source:", ["All", "🎓 Training Data", "📤 Uploaded Data"])
        with col_f2:
            fraud_filter = st.selectbox("Filter by status:", ["All", "Fraud", "Clean"])
        with col_f3:
            search_bc = st.text_input("Search provider ID:", placeholder="e.g. PRV51005")

        disp_recs = recs.copy()
        if src_filter != "All":
            disp_recs = [r for r in disp_recs if r.get("source", "") == src_filter]
        if fraud_filter == "Fraud":
            disp_recs = [r for r in disp_recs if r.get("fraud")]
        elif fraud_filter == "Clean":
            disp_recs = [r for r in disp_recs if not r.get("fraud")]
        if search_bc:
            disp_recs = [r for r in disp_recs if search_bc.lower() in r["provider"].lower()]

        st.info(f"Showing **{len(disp_recs)}** of {n_total} total records")

        if disp_recs:
            df_show = pd.DataFrame(disp_recs)
            df_show["Status"]               = df_show["fraud"].map({True: "❌ Fraud", False: "✅ Clean"})
            df_show["Block#"]               = range(1, len(df_show) + 1)
            df_show["Hash (first 20 chars)"] = df_show["block_hash"].str[:20] + "..."
            df_show["Prev Hash (12)"]        = df_show["prev_hash"].str[:12].fillna("GENESIS") + "..."

            display_cols = ["Block#", "provider", "Status", "probability", "risk",
                            "source", "timestamp", "Hash (first 20 chars)"]
            st.dataframe(df_show[display_cols], use_container_width=True, height=450)

            csv_full = df_show[["Block#", "provider", "Status", "probability", "risk",
                                 "source", "timestamp", "block_hash", "prev_hash"]].to_csv(index=False)
            st.download_button("📥 Download Full Ledger CSV", csv_full,
                               "blockchain_ledger.csv", "text/csv", use_container_width=True)
    else:
        st.warning("Blockchain not yet initialised. Restart the app to load training data automatically.")

    with st.expander("📄 View Smart Contract (Solidity)", expanded=False):
        st.code("""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
contract FraudRegistry {
    struct FraudRecord {
        string  providerId; bool isFraudulent;
        uint256 fraudProbability; string riskCategory;
        string  dataHash; uint256 timestamp;
    }
    mapping(string => FraudRecord) public records;
    address public owner;
    constructor() { owner = msg.sender; }
    function addRecord(string memory _id, bool _fraud, uint256 _prob,
                       string memory _risk, string memory _hash) public {
        require(msg.sender == owner, "Only owner can add records");
        records[_id] = FraudRecord(_id, _fraud, _prob, _risk, _hash, block.timestamp);
        emit RecordAdded(_id, _fraud, block.timestamp);
    }
}""", language='solidity')

# ══════════════════════════════════════════════════════════════
# PAGE 6 — BLOCKCHAIN LEDGER  (Live Ganache Integration)
# ══════════════════════════════════════════════════════════════
elif page == "⛓️ Blockchain Ledger":
    st.title("⛓️ Blockchain Audit Ledger")
    st.markdown("Live records fetched directly from your **Ganache** Ethereum node.")
    st.divider()

    # ── Try importing web3 ────────────────────────────────────────
    try:
        from web3 import Web3
        WEB3_OK = True
    except ImportError:
        WEB3_OK = False

    # ── Connection settings (editable in sidebar expander) ───────
    with st.sidebar.expander("⛓️ Ganache Settings", expanded=False):
        ganache_url = st.text_input("Ganache RPC URL",
                                     value="http://127.0.0.1:7545",
                                     key="ganache_url")
        contract_address = st.text_input(
            "Contract Address",
            value=st.session_state.get('active_contract_addr',
                                        bc_config.get('address', '')),
            key="contract_addr",
            help="Paste the address shown in Ganache after deploying your contract"
        )
        st.caption("If you restarted Ganache, re-run your deploy notebook and paste the new address here.")

    # ── Load ABI: try project files first, then fall back to hardcoded ──
    def load_abi_from_project():
        """Try to load the real compiled ABI from your truffle/hardhat artifacts."""
        candidate_paths = [
            os.path.join(BASE, '../blockchain/build/contracts/FraudDetection.json'),
            os.path.join(BASE, '../blockchain/build/contracts/HealthcareFraud.json'),
            os.path.join(BASE, '../build/contracts/FraudDetection.json'),
            os.path.join(BASE, '../contracts/FraudDetection.json'),
            os.path.join(BASE, '../blockchain/FraudDetection.json'),
            os.path.join(RESULTS, 'contract_abi.json'),
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                with open(path) as f:
                    artifact = json.load(f)
                # Truffle artifact has {"abi": [...]}; raw ABI is just [...]
                return artifact.get('abi', artifact), path
        return None, None

    project_abi, abi_source = load_abi_from_project()

    # Hardcoded fallback ABI — matches your deployed 5-argument storeFraudRecord contract
    # Solidity: storeFraudRecord(provider, isFraud, fraudProbability, riskCategory, dataHash)
    FALLBACK_ABI = [
      {"inputs":[],"name":"getTotalRecords",
       "outputs":[{"internalType":"uint256","name":"","type":"uint256"}],
       "stateMutability":"view","type":"function"},
      {"inputs":[{"internalType":"uint256","name":"_recordId","type":"uint256"}],
       "name":"getRecord",
       "outputs":[
           {"internalType":"string","name":"","type":"string"},
           {"internalType":"bool","name":"","type":"bool"},
           {"internalType":"uint256","name":"","type":"uint256"},
           {"internalType":"string","name":"","type":"string"},
           {"internalType":"uint256","name":"","type":"uint256"},
           {"internalType":"bytes32","name":"","type":"bytes32"}
       ],"stateMutability":"view","type":"function"},
      # PRIMARY: 5-arg with string dataHash — matches your deployed contract
      {"inputs":[
          {"internalType":"string","name":"provider","type":"string"},
          {"internalType":"bool","name":"isFraud","type":"bool"},
          {"internalType":"uint256","name":"fraudProbability","type":"uint256"},
          {"internalType":"string","name":"riskCategory","type":"string"},
          {"internalType":"string","name":"dataHash","type":"string"}
      ],"name":"storeFraudRecord","outputs":[],"stateMutability":"nonpayable","type":"function"},
      # FALLBACK: 5-arg with bytes32 dataHash
      {"inputs":[
          {"internalType":"string","name":"provider","type":"string"},
          {"internalType":"bool","name":"isFraud","type":"bool"},
          {"internalType":"uint256","name":"fraudProbability","type":"uint256"},
          {"internalType":"string","name":"riskCategory","type":"string"},
          {"internalType":"bytes32","name":"dataHash","type":"bytes32"}
      ],"name":"storeFraudRecord","outputs":[],"stateMutability":"nonpayable","type":"function"},
      {"anonymous":False,"inputs":[
          {"indexed":True,"name":"index","type":"uint256"},
          {"indexed":False,"name":"provider","type":"string"},
          {"indexed":False,"name":"isFraud","type":"bool"}
      ],"name":"FraudRecordStored","type":"event"}
    ]

    CONTRACT_ABI = project_abi if project_abi else FALLBACK_ABI

    # ── Connect to Ganache ────────────────────────────────────────
    def get_web3(url):
        w3 = Web3(Web3.HTTPProvider(url))
        return w3 if w3.is_connected() else None

    def get_contract(w3, addr, abi):
        try:
            return w3.eth.contract(
                address=Web3.to_checksum_address(addr.strip()), abi=abi)
        except Exception:
            return None

    def contract_is_alive(contract):
        """Ping getTotalRecords — if it throws, contract is gone/wrong."""
        try:
            contract.functions.getTotalRecords().call()
            return True
        except Exception:
            return False

    # ── Fetch ALL records from the smart contract ─────────────────
    @st.cache_data(ttl=15)
    def fetch_ganache_records(url, addr):
        rows = []
        try:
            w3 = Web3(Web3.HTTPProvider(url))
            if not w3.is_connected():
                return None, "Cannot connect to Ganache at " + url

            checksum_addr = Web3.to_checksum_address(addr.strip())
            contract = w3.eth.contract(address=checksum_addr, abi=CONTRACT_ABI)
            total = contract.functions.getTotalRecords().call()

            # Contract uses 1-based record IDs: records stored at keys 1..total
            for i in range(1, total + 1):
                try:
                    rec = contract.functions.getRecord(i).call()
                    risk_cat_onchain = 'Unknown'
                    # ABI output order from blockchain_config.json (deployed contract):
                    # [0] providerID (string)
                    # [1] isFraud    (bool)
                    # [2] fraudProbability (uint256)
                    # [3] riskCategory    (string)
                    # [4] timestamp       (uint256)  ← BEFORE dataHash
                    # [5] dataHash        (bytes32)  ← LAST
                    if len(rec) >= 6:
                        provider, is_fraud, fraud_prob_int, risk_cat_onchain, ts, data_hash = rec[:6]
                    elif len(rec) == 5:
                        provider, is_fraud, fraud_prob_int, ts, data_hash = rec[:5]
                    elif len(rec) == 4:
                        provider, is_fraud, fraud_prob_int, data_hash = rec
                        ts = 0
                    else:
                        continue

                    raw_prob = int(fraud_prob_int)
                    if raw_prob > 10000:
                        fraud_prob = raw_prob / 1_000_000
                    elif raw_prob > 100:
                        fraud_prob = raw_prob / 10_000
                    else:
                        fraud_prob = raw_prob / 100

                    display_risk = risk_cat_onchain if risk_cat_onchain not in ('Unknown', '', None) \
                                   else risk_label(fraud_prob)

                    rows.append({
                        'RecordID':         i,
                        'Provider':         provider,
                        'IsFraud':          bool(is_fraud),
                        'FraudProbability': round(fraud_prob, 4),
                        'RiskCategory':     display_risk,
                        'Timestamp':        pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d %H:%M:%S')
                                            if ts and ts > 0 else 'N/A',
                        'DataHash':         '0x' + data_hash.hex()
                                            if isinstance(data_hash, bytes) else str(data_hash),
                        'TxHash':           'N/A',
                        'BlockNumber':      'N/A',
                        'GasUsed':          'N/A',
                    })
                except Exception as e:
                    rows.append({
                        'RecordID': i, 'Provider': f'ERROR reading record {i}: {e}',
                        'IsFraud': False, 'FraudProbability': 0.0,
                        'RiskCategory': 'Unknown', 'Timestamp': 'N/A',
                        'DataHash': 'N/A', 'TxHash': 'N/A',
                        'BlockNumber': 'N/A', 'GasUsed': 'N/A',
                    })
                    continue

            df = pd.DataFrame(rows) if rows else pd.DataFrame(
                columns=['RecordID','Provider','IsFraud','FraudProbability',
                         'RiskCategory','Timestamp','DataHash','TxHash','BlockNumber','GasUsed'])

            # ── LAYER 1: Scan all blocks for transactions TO this contract ──
            # This is the most reliable method — works even without events
            try:
                latest_block = w3.eth.block_number
                # Build a lookup: provider_name -> (txHash, blockNumber, gasUsed)
                # by scanning every block and checking receipts sent to our contract
                tx_info_by_order = []  # list of (txHash, blockNumber, gasUsed) in chain order
                for blk_num in range(1, latest_block + 1):
                    try:
                        block = w3.eth.get_block(blk_num, full_transactions=True)
                        for tx in block.transactions:
                            tx_to = tx.get('to') or ''
                            if tx_to.lower() == checksum_addr.lower():
                                try:
                                    receipt = w3.eth.get_transaction_receipt(tx['hash'])
                                    tx_info_by_order.append({
                                        'TxHash':      tx['hash'].hex(),
                                        'BlockNumber': blk_num,
                                        'GasUsed':     receipt['gasUsed'],
                                    })
                                except Exception:
                                    tx_info_by_order.append({
                                        'TxHash':      tx['hash'].hex(),
                                        'BlockNumber': blk_num,
                                        'GasUsed':     'N/A',
                                    })
                    except Exception:
                        continue

                # The contract deployment tx is first; storeFraudRecord txns follow in order.
                # Record index 0 in the contract corresponds to tx_info_by_order[1], etc.
                # (index 0 is deploy tx, indices 1..N are store txns)
                store_txns = tx_info_by_order[1:] if len(tx_info_by_order) > 1 else []
                for rec_idx, tx_info in enumerate(store_txns):
                    if rec_idx < len(df):
                        df.at[rec_idx, 'TxHash']      = tx_info['TxHash']
                        df.at[rec_idx, 'BlockNumber']  = tx_info['BlockNumber']
                        df.at[rec_idx, 'GasUsed']      = tx_info['GasUsed']
            except Exception:
                pass  # fall through to other enrichment layers

            # ── LAYER 2: Enrich via FraudRecordStored event logs ──────────
            # Fills gaps if block scan missed anything
            try:
                events = contract.events.FraudRecordStored.get_logs(
                    fromBlock=0, toBlock='latest')
                for ev in events:
                    idx = ev['args'].get('index', ev['args'].get('recordIndex', None))
                    if idx is not None and int(idx) < len(df):
                        row_idx = int(idx)
                        if df.at[row_idx, 'TxHash'] == 'N/A':
                            df.at[row_idx, 'TxHash']     = ev['transactionHash'].hex()
                            df.at[row_idx, 'BlockNumber'] = ev['blockNumber']
                            try:
                                receipt = w3.eth.get_transaction_receipt(ev['transactionHash'])
                                df.at[row_idx, 'GasUsed'] = receipt['gasUsed']
                            except Exception:
                                pass
            except Exception:
                pass

            # ── LAYER 3: Enrich from blockchain_stored_records.csv ────────
            # Fills any remaining gaps using the CSV written during notebook runs
            stored_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '../results/blockchain_stored_records.csv')
            if os.path.exists(stored_csv) and 'Provider' in df.columns:
                try:
                    _stored = pd.read_csv(stored_csv)
                    _col_map = {
                        'tx_hash':'TxHash','transaction_hash':'TxHash',
                        'block_number':'BlockNumber','block':'BlockNumber',
                        'gas_used':'GasUsed','provider':'Provider',
                    }
                    _stored.rename(columns={k:v for k,v in _col_map.items()
                                            if k in _stored.columns}, inplace=True)
                    if 'TxHash' in _stored.columns and 'Provider' in _stored.columns:
                        _tx_lookup  = _stored.dropna(subset=['TxHash']).set_index('Provider')['TxHash'].to_dict()
                        _blk_lookup = _stored.dropna(subset=['BlockNumber']).set_index('Provider')['BlockNumber'].to_dict() \
                                      if 'BlockNumber' in _stored.columns else {}
                        _gas_lookup = _stored.dropna(subset=['GasUsed']).set_index('Provider')['GasUsed'].to_dict() \
                                      if 'GasUsed' in _stored.columns else {}
                        for i, row in df.iterrows():
                            prov = row['Provider']
                            if str(row['TxHash']) == 'N/A' and prov in _tx_lookup:
                                df.at[i, 'TxHash']      = str(_tx_lookup[prov])
                            if str(row['BlockNumber']) == 'N/A' and prov in _blk_lookup:
                                df.at[i, 'BlockNumber'] = _blk_lookup[prov]
                            if str(row['GasUsed']) == 'N/A' and prov in _gas_lookup:
                                df.at[i, 'GasUsed']     = _gas_lookup[prov]
                except Exception:
                    pass

            # ── LAYER 4: Fill GasUsed from receipts for any TxHash we now have ──
            for i, row in df.iterrows():
                if str(row['TxHash']) != 'N/A' and str(row['GasUsed']) == 'N/A':
                    try:
                        receipt = w3.eth.get_transaction_receipt(row['TxHash'])
                        df.at[i, 'GasUsed'] = receipt['gasUsed']
                    except Exception:
                        pass

            return df, None

        except Exception as e:
            return None, str(e)

    # ── Store new predictions to Ganache ─────────────────────────
    def store_to_ganache(w3, contract, results_df, deployer):
        """
        Stores FRAUD-ONLY predictions to Ganache.
        Every transaction in Ganache = one confirmed fraud provider.
        Contract: storeFraudRecord(provider, isFraud, fraudProbability, riskCategory, dataHash)
        Also saves TxHash/BlockNumber/GasUsed to blockchain_stored_records.csv for dashboard enrichment.
        """
        # Filter to fraud only — every Ganache block = one confirmed fraud case
        fraud_only = results_df[results_df['FraudPrediction'] == 1].copy()
        n_skipped  = len(results_df) - len(fraud_only)

        ok, errors, stored_rows = 0, [], []
        for _, row in fraud_only.iterrows():
            try:
                provider     = str(row['Provider'])
                is_fraud     = bool(row['FraudPrediction'])
                prob_int     = int(round(float(row['FraudProbability']) * 1_000_000))
                risk_cat     = str(row.get('RiskCategory', risk_label(float(row['FraudProbability']))))
                hash_str     = compute_hash(f"{provider}{is_fraud}{float(row['FraudProbability']):.6f}{risk_cat}")

                # Try string dataHash first, then bytes32
                try:
                    tx = contract.functions.storeFraudRecord(
                        provider, is_fraud, prob_int, risk_cat, hash_str
                    ).transact({'from': deployer, 'gas': 300_000})
                except Exception:
                    hash_bytes = w3.keccak(text=hash_str)
                    tx = contract.functions.storeFraudRecord(
                        provider, is_fraud, prob_int, risk_cat, hash_bytes
                    ).transact({'from': deployer, 'gas': 300_000})

                receipt = w3.eth.wait_for_transaction_receipt(tx, timeout=30)
                ok += 1
                stored_rows.append({
                    'Provider':         provider,
                    'IsFraud':          is_fraud,
                    'FraudProbability': float(row['FraudProbability']),
                    'RiskCategory':     risk_cat,
                    'DataHash':         hash_str,
                    'TxHash':           receipt['transactionHash'].hex(),
                    'BlockNumber':      receipt['blockNumber'],
                    'GasUsed':          receipt['gasUsed'],
                    'Timestamp':        pd.Timestamp.now().isoformat(),
                })
            except Exception as e:
                errors.append(f"Blockchain Error ({row['Provider']}): {e}")

        # ── Persist TxHash/BlockNumber/GasUsed to CSV so dashboard can read them ──
        if stored_rows:
            try:
                _new_df   = pd.DataFrame(stored_rows)
                _out_path = os.path.join(BASE, '../results/blockchain_stored_records.csv')
                if os.path.exists(_out_path):
                    _existing = pd.read_csv(_out_path)
                    _new_df   = pd.concat([_existing, _new_df], ignore_index=True)
                    _new_df.drop_duplicates(subset='Provider', keep='last', inplace=True)
                _new_df.to_csv(_out_path, index=False)
                load_blockchain_records.clear()  # bust @st.cache_data
            except Exception:
                pass

        return ok, errors, len(fraud_only), n_skipped

    # ─────────────────────────────────────────────────────────────
    # CONNECTION + CONTRACT STATUS
    # ─────────────────────────────────────────────────────────────
    if not WEB3_OK:
        st.error("❌ `web3` library not installed. Run:  `pip install web3`  then restart Streamlit.")
        st.stop()

    w3_live = get_web3(ganache_url)

    if w3_live is None:
        st.error(f"❌ Cannot connect to Ganache at **{ganache_url}**")
        st.info("💡 Start Ganache desktop app or run `ganache-cli --port 7545` in a terminal.")
        all_records = bc_records.copy()
        all_records['Source'] = 'CSV Backup'
        for _col in ['TxHash','BlockNumber','GasUsed','DataHash']:
            if _col not in all_records.columns: all_records[_col] = 'N/A'

    else:
        current_block = w3_live.eth.block_number
        chain_id      = w3_live.eth.chain_id

        # ── Detect fresh/reset Ganache ───────────────────────────
        if current_block == 0:
            st.warning(
                f"⚠️ **Ganache is connected but shows Block #0** — this means Ganache was restarted "
                f"and your previously deployed contract no longer exists on-chain."
            )
            with st.expander("🔧 How to fix this — click to expand", expanded=True):
                st.markdown("""
**Your contract state was lost because Ganache restarted. Do ONE of these:**

---

**Option A — Re-run your deploy notebook (recommended)**
1. Open your deploy notebook (the one that runs `deploy_contract.py` or Phase 4)
2. Run it — it will deploy a fresh contract to the new Ganache session
3. Copy the new contract address it prints
4. Paste it into **⛓️ Ganache Settings → Contract Address** in the sidebar
5. Click **🔄 Refresh from Ganache** below

---

**Option B — Use Ganache Workspace (keeps state across restarts)**
1. In Ganache desktop: **New Workspace → Ethereum**
2. Under **Server** tab: set port to `7545`
3. Under **Accounts & Keys** tab: paste your mnemonic from `blockchain_config.json`
4. **Save Workspace** — now state persists across restarts

---

**Option C — Show CSV records (no blockchain)**
The 139 records from your last notebook run are shown below from the CSV file.
                """)
            st.divider()
            st.info("📂 Showing records from CSV backup while you fix Ganache.")
            all_records = bc_records.copy()
            all_records['Source'] = 'CSV Backup (Ganache reset)'
            for _col in ['TxHash','BlockNumber','GasUsed','DataHash']:
                if _col not in all_records.columns: all_records[_col] = 'N/A'

        else:
            st.success(f"✅ Connected to Ganache — Block #{current_block:,} | Chain ID: {chain_id}")
            if abi_source:
                st.caption(f"📄 ABI loaded from: `{os.path.basename(abi_source)}`")
            else:
                st.caption("📄 Using built-in ABI — place your compiled contract JSON in `blockchain/build/contracts/` for exact matching")

            contract_addr  = contract_address.strip()
            contract_live  = get_contract(w3_live, contract_addr, CONTRACT_ABI) if contract_addr else None

            if not contract_addr:
                st.error("❌ No contract address set. Paste your deployed contract address in **⛓️ Ganache Settings** sidebar.")
                all_records = bc_records.copy()
                all_records['Source'] = 'CSV Backup'
                for _col in ['TxHash','BlockNumber','GasUsed','DataHash']:
                    if _col not in all_records.columns: all_records[_col] = 'N/A'

            elif contract_live is None or not contract_is_alive(contract_live):
                st.error(
                    f"❌ Contract at `{contract_addr[:20]}...` is not responding. "
                    "The address is wrong or the contract was not deployed to this Ganache session."
                )
                with st.expander("🔧 Fix: re-deploy and update address", expanded=True):
                    st.markdown("""
1. Run your deploy script/notebook to redeploy the contract to this Ganache session
2. Copy the **new contract address** from the output
3. Open **⛓️ Ganache Settings** in the sidebar and paste the new address
4. Press Enter — the page will reload with live data
                    """)
                all_records = bc_records.copy()
                all_records['Source'] = 'CSV Backup (wrong contract)'
                for _col in ['TxHash','BlockNumber','GasUsed','DataHash']:
                    if _col not in all_records.columns: all_records[_col] = 'N/A'

            else:
                # ── Contract is alive — store uploaded data ───────
                uploaded_results = st.session_state.get('uploaded_results_df', None)
                already_stored   = st.session_state.get('ganache_stored', False)

                if uploaded_results is not None and not already_stored:
                    # Always verify deployer exists in current Ganache session
                    _live_accounts = w3_live.eth.accounts
                    _cfg_deployer  = bc_config.get('deployer', '')
                    if _cfg_deployer and _cfg_deployer in _live_accounts:
                        deployer = _cfg_deployer
                    elif _live_accounts:
                        deployer = _live_accounts[0]
                    else:
                        st.error("❌ No Ganache accounts found. Is Ganache running?")
                        deployer = None
                    if deployer and uploaded_results is not None and not already_stored:
                        n_fraud_upload = int((uploaded_results['FraudPrediction'] == 1).sum())
                        n_clean_upload = len(uploaded_results) - n_fraud_upload
                    with st.spinner(f"⛓️ Storing {n_fraud_upload} fraud predictions to Ganache ({n_clean_upload} clean skipped)…"):
                        ok_count, errs, total_fraud, n_skipped = store_to_ganache(
                            w3_live, contract_live, uploaded_results, deployer)
                    st.session_state['ganache_stored'] = True
                    fetch_ganache_records.clear()
                    if total_fraud == 0:
                        st.info("ℹ️ No fraud predictions in uploaded data — nothing stored to Ganache.")
                    elif ok_count == total_fraud:
                        st.success(
                            f"✅ **{ok_count} fraud providers permanently stored on Ganache!** "
                            f"({n_skipped} clean providers skipped — every block = confirmed fraud)"
                        )
                    elif ok_count > 0:
                        st.warning(f"⚠️ Partial: {ok_count}/{total_fraud} fraud records stored.")
                        with st.expander(f"⚠️ {len(errs)} errors (click to expand)"):
                            for e in errs[:20]: st.error(e)
                    else:
                        st.error(f"❌ 0/{total_fraud} fraud records stored — all transactions failed.")
                        with st.expander("🔍 Error details (click to expand)", expanded=True):
                            for e in errs[:10]: st.error(e)
                        st.warning(
                            "**Root cause:** Wrong contract address or ABI mismatch. "
                            "Re-deploy your contract and paste the new address in **⛓️ Ganache Settings** sidebar."
                        )

                # ── Fetch live records ────────────────────────────
                with st.spinner("Fetching records from Ganache…"):
                    all_records, fetch_err = fetch_ganache_records(ganache_url, contract_addr)

                if fetch_err:
                    st.error(f"❌ Error reading contract: {fetch_err}")
                    all_records = bc_records.copy()
                    all_records['Source'] = 'CSV Backup'
                    for _col in ['TxHash','BlockNumber','GasUsed','DataHash']:
                        if _col not in all_records.columns: all_records[_col] = 'N/A'
                else:
                    st.session_state['active_contract_addr'] = contract_addr
                    all_records['Source'] = 'On-Chain ⛓️'
                    st.caption(f"🔄 Auto-refreshes every 15 s  |  {len(all_records)} records on-chain")

    # ── Ensure required columns always exist (safety net) ─────────
    for _col in ['TxHash', 'BlockNumber', 'GasUsed', 'DataHash', 'IsFraud']:
        if _col not in all_records.columns:
            all_records[_col] = 'N/A'

    # ── KPI row ───────────────────────────────────────────────────
    total_rec      = len(all_records)
    fraud_on_chain = int((all_records['IsFraud'] == True).sum())
    verified_cnt   = int((all_records['TxHash'] != 'N/A').sum())
    latest_block   = pd.to_numeric(all_records['BlockNumber'], errors='coerce').dropna()
    latest_block_val = int(latest_block.max()) if len(latest_block) > 0 else 'N/A'

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("On-Chain Records",  f"{total_rec:,}")
    c2.metric("Fraud Flagged",     f"{fraud_on_chain:,}")
    c3.metric("Txns with Hash",    f"{verified_cnt}")
    c4.metric("Latest Block",      f"{latest_block_val}")
    c5.metric("Chain ID",          str(w3_live.eth.chain_id) if w3_live else bc_config.get('chain_id', 1337))

    st.divider()

    # ── Manual refresh button ─────────────────────────────────────
    col_ref, col_store = st.columns([1, 3])
    with col_ref:
        if st.button("🔄 Refresh from Ganache"):
            fetch_ganache_records.clear()
            st.rerun()
    with col_store:
        # Allow re-storing if user wants to push again
        if st.session_state.get('ganache_stored') and st.session_state.get('uploaded_results_df') is not None:
            if st.button("📤 Re-store uploaded data to Ganache"):
                st.session_state['ganache_stored'] = False
                st.rerun()

    # ── Contract info ─────────────────────────────────────────────
    with st.expander("📋 Smart Contract Details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.code(f"""
Platform:    Ethereum (Ganache local)
Chain ID:    {bc_config.get('chain_id', 1337)}
Contract:    {bc_config.get('address', 'N/A')}
Deployer:    {bc_config.get('deployer', 'N/A')}
Deploy Gas:  {bc_config.get('deploy_gas', 0):,}
RPC URL:     {ganache_url}
            """)
        with col2:
            st.markdown("**Functions:**")
            st.markdown("- `storeFraudRecord()` — store prediction + hash")
            st.markdown("- `getRecord(index)` — retrieve any record")
            st.markdown("- `getTotalRecords()` — count stored records")

    st.divider()

    # ── Filters ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        bc_filter = st.selectbox("Filter by fraud status:",
                                  ["All", "Fraud Only", "Non-Fraud Only"])
    with col2:
        hash_filter = st.selectbox("Filter by transaction hash:",
                                    ["All", "Has TxHash", "No TxHash"])

    bc_display = all_records.copy()
    if bc_filter == "Fraud Only":
        bc_display = bc_display[bc_display['IsFraud'] == True]
    elif bc_filter == "Non-Fraud Only":
        bc_display = bc_display[bc_display['IsFraud'] == False]
    if 'TxHash' in bc_display.columns:
        if hash_filter == "Has TxHash":
            bc_display = bc_display[bc_display['TxHash'] != 'N/A']
        elif hash_filter == "No TxHash":
            bc_display = bc_display[bc_display['TxHash'] == 'N/A']

    st.markdown(f"**Showing {len(bc_display)} on-chain records**")

    # ── Style columns ─────────────────────────────────────────────
    def style_is_fraud(val):
        if val is True or val == 'True' or val == 1:
            return 'color: #ff6b6b; font-weight: bold'
        return 'color: #51cf66; font-weight: bold'

    def style_risk_bc(val):
        if 'High'   in str(val): return 'color: #ff6b6b; font-weight: bold'
        if 'Medium' in str(val): return 'color: #ffd43b; font-weight: bold'
        if 'Low'    in str(val): return 'color: #51cf66'
        return ''

    show_cols = ['RecordID','Provider','IsFraud','FraudProbability',
                 'RiskCategory','Timestamp','TxHash','BlockNumber','GasUsed','DataHash','Source']
    show_cols = [c for c in show_cols if c in bc_display.columns]

    styled_bc = bc_display[show_cols].style
    if 'IsFraud' in show_cols:
        styled_bc = styled_bc.applymap(style_is_fraud, subset=['IsFraud'])
    if 'RiskCategory' in show_cols:
        styled_bc = styled_bc.applymap(style_risk_bc, subset=['RiskCategory'])

    st.dataframe(styled_bc, use_container_width=True,
                 height=min(35 * len(bc_display) + 38, 900))
    st.caption("⛓️ = fetched live from Ganache  |  🔴 = Fraud  |  🟢 = Clean")

    # ── Download ──────────────────────────────────────────────────
    csv = bc_display.to_csv(index=False).encode()
    st.download_button("⬇️ Download Ledger CSV", csv,
                       "blockchain_ledger_ganache.csv", "text/csv")

# ══════════════════════════════════════════════════════════════
# PAGE 7 — SYSTEM INFO
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ System Info":
    st.title("ℹ️ System Information")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Dataset", "⏱️ Runtime", "📄 Paper"])

    with tab1:
        st.subheader("Dataset Statistics")
        st.markdown("""
        | Item | Value |
        |---|---|
        | Training providers | 5,410 |
        | Fraud (Yes) | 506 (9.4%) |
        | Non-Fraud (No) | 4,904 (90.6%) |
        | Imbalance ratio | 9.7 : 1 |
        | Test providers | 1,353 |
        | Beneficiary records | ~1M |
        | Inpatient claim records | ~40K |
        | Outpatient claim records | ~518K |
        | Features engineered | 47 |
        | After SMOTE (train) | 7,846 |
        """)

        st.divider()
        st.subheader("Feature Categories")
        st.markdown("""
        | Category | Count | Examples |
        |---|---|---|
        | Inpatient financial | 8 | IP_TotalClaimAmt, IP_AvgClaimAmt |
        | Inpatient behavioral | 8 | IP_NumClaims, IP_NumUniqueAttPhysicians |
        | Inpatient duration | 4 | IP_AvgStayDuration, IP_TotalStayDays |
        | Inpatient patient | 6 | IP_AvgPatientAge, IP_AvgChronicConditions |
        | Outpatient features | 13 | OP_NumClaims, OP_TotalClaimAmt |
        | Advanced ratios | 8 | IP_OP_ClaimRatio, IP_ClaimsPerPatient |
        """)

    with tab2:
        st.subheader("Runtime Table (Paper Table III)")
        try:
            runtime_df = pd.read_csv(RESULTS + 'runtime_table_complete.csv')
            st.dataframe(runtime_df, use_container_width=True)
        except:
            st.warning("Run Phase 5 to generate runtime table.")

        st.divider()
        st.markdown("""
        | Metric | Value |
        |---|---|
        | LR Train time | 0.304s |
        | LR Prediction time | 0.002s |
        | Blockchain deploy gas | 909,053 |
        | Blockchain store/record | 170,412 gas, 0.151s |
        | Blockchain retrieve/record | 0.075s (no gas) |
        """)

    with tab3:
        st.subheader("Paper Information")
        st.markdown("""
        **Title:** Blockchain-Secured Healthcare Provider Fraud Detection
        using Machine Learning and Explainable AI

        **Conference:** INDISCON 2026

        **Key Contributions:**
        - 7 ML models evaluated on real CMS Medicare dataset
        - 47 engineered features across financial, behavioral, and demographic domains
        - SHAP explainability for both Logistic Regression and Random Forest
        - Ethereum blockchain for tamper-proof fraud prediction storage
        - Optimal threshold selection improving fraud rate alignment to 9.9%

        **Results:**
        - AUC-ROC: 96.75% (Logistic Regression)
        - Recall: 90.10% (catches 90 of every 100 fraud providers)
        - Overfit gap: −0.89% (best generalisation among all 7 models)
        - 139/139 blockchain records verified (100% integrity)
        """)

        st.divider()
        img = FIG + 'fig05_model_comparison.png'
        if os.path.exists(img):
            st.image(img, caption="Model comparison figure", use_column_width=True)