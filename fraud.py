import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Optimization Engine", layout="wide")

# --- 1. UNIFIED DATA GENERATOR & LOADER ---
@st.cache_data
def get_or_create_data():
    np.random.seed(42) # Fixed seed so the demo is stable
    num_records = 30000
    markets = ['FO_NO', 'FO_SE', 'FO_FI', 'FO_DK', 'FO_EE']
    entity_col = np.random.choice(markets, num_records, p=[0.25, 0.25, 0.20, 0.15, 0.15])
    dates = pd.date_range(start='2026-01-01', periods=num_records, freq='5T')
    amount_eur = np.random.lognormal(mean=3.0, sigma=0.5, size=num_records).round(2)

    vendor_vertical = np.random.choice(['restaurants', 'supermarket', 'darkstores'], num_records, p=[0.6, 0.3, 0.1])
    has_discount = np.random.choice([True, False], num_records, p=[0.2, 0.8])
    customer_source_type = np.random.choice(['New', 'Existing'], num_records, p=[0.25, 0.75])
    is_saved_card = np.where(customer_source_type == 'Existing', np.random.choice([True, False], num_records, p=[0.8, 0.2]), False)
    issuer_country_group = np.random.choice(['Local', 'International'], num_records, p=[0.85, 0.15])
    is_3ds_eligible = np.random.choice([True, False], num_records, p=[0.9, 0.1])

    # Risk Score Generation
    risk_score = np.random.beta(a=2, b=6, size=num_records) * 100
    risk_score = np.where(entity_col == 'FO_EE', risk_score + 15, risk_score)
    risk_score = np.where(entity_col == 'FO_NO', risk_score - 10, risk_score)
    risk_score = np.clip(risk_score, 0, 99).round(2)

    # Actual Fraud (Ground Truth)
    prob = (risk_score / 100) ** 1.5 
    prob = np.where(issuer_country_group == 'International', prob * 1.5, prob)
    is_actual_chargeback = np.random.binomial(1, np.clip(prob, 0, 1))

    # Rule Assignment based on Market Context
    rule_triggered = np.array(['No_Rule'] * num_records, dtype=object)
    for i in range(num_records):
        mkt = entity_col[i]
        score = risk_score[i]
        if mkt == 'FO_NO': # Too Strict
            if score > 30: rule_triggered[i] = 'velocity_high_risk_block'
            elif score > 15: rule_triggered[i] = 'card_account_age_7d_3ds'
        elif mkt == 'FO_SE': # Optimal
            if score > 60: rule_triggered[i] = 'velocity_high_risk_block'
            elif score > 40: rule_triggered[i] = 'card_limit_value_3ds'
        elif mkt == 'FO_EE': # Too Loose
            if score > 80: rule_triggered[i] = 'foreign_card_review'
        elif mkt == 'FO_FI':
            if score > 50: rule_triggered[i] = 'card_limit_value_3ds'
            elif score > 25: rule_triggered[i] = 'foreign_card_review'
        else: # FO_DK
            if score > 65: rule_triggered[i] = 'card_account_age_7d_3ds'

    rule_action = np.where(pd.Series(rule_triggered).str.contains('block'), 'hard_block', 
                  np.where(pd.Series(rule_triggered) == 'No_Rule', 'none', 'review'))

    # User's underlying propensity to switch payments if blocked
    switch_propensity = np.random.binomial(1, 0.25, num_records)
    payment_switch = np.zeros(num_records)
    order_final_status = np.array(['DELIVERED'] * num_records, dtype=object)

    for i in range(num_records):
        if rule_triggered[i] != 'No_Rule':
            if rule_action[i] == 'hard_block':
                order_final_status[i] = 'CANCELLED'
                if is_actual_chargeback[i] == 0: payment_switch[i] = switch_propensity[i]
            elif rule_action[i] == 'review':
                if is_actual_chargeback[i] == 1:
                    order_final_status[i] = 'CANCELLED'
                else:
                    drop_rate = 0.8 if not is_3ds_eligible[i] else 0.3
                    if np.random.rand() < drop_rate:
                        order_final_status[i] = 'CANCELLED'
                        payment_switch[i] = switch_propensity[i]

    df = pd.DataFrame({
        'order_date': dates,
        'Month': dates.strftime('%Y-%m'),
        'global_entity_id': entity_col,
        'amount_eur': amount_eur,
        'vendor_vertical': vendor_vertical,
        'is_saved_card': is_saved_card,
        'rule_name': rule_triggered,
        'rule_action': rule_action,
        'risk_score': risk_score,
        'switch_propensity': switch_propensity, 
        'payment_switch': payment_switch,
        'order_final_status': order_final_status,
        'is_actual_chargeback': is_actual_chargeback
    })
    return df

df_raw = get_or_create_data()

# --- 2. SIDEBAR FILTERS ---
st.sidebar.header("1. Global Filters")
if st.sidebar.button("🔄 Clear Cache & Regenerate"):
    st.cache_data.clear()
    st.rerun()

market_opts = ['All Markets'] + sorted(list(df_raw['global_entity_id'].unique()))
market = st.sidebar.selectbox("Market Entity", options=market_opts)

month_opts = ['All Months'] + sorted(list(df_raw['Month'].unique()))
month = st.sidebar.selectbox("Month", options=month_opts)

vertical_opts = ['All Verticals'] + sorted(list(df_raw['vendor_vertical'].unique()))
vertical = st.sidebar.selectbox("Vendor Vertical", options=vertical_opts)

card_opts = ['All Cards', 'Saved Card Only', 'New Card Only']
is_saved = st.sidebar.selectbox("Card Status", options=card_opts)

st.sidebar.header("2. Economic Assumptions")
margin_pct = st.sidebar.slider("Net Profit Margin (%)", 1, 30, 10) / 100.0
cb_fee = st.sidebar.number_input("Chargeback Fee (€)", value=15.0)
ltv_cost = st.sidebar.number_input("Blended LTV Cost per Churn (€)", value=50.0)

# --- APPLY FILTERS ---
df = df_raw.copy()
if market != 'All Markets': df = df[df['global_entity_id'] == market]
if month != 'All Months': df = df[df['Month'] == month]
if vertical != 'All Verticals': df = df[df['vendor_vertical'] == vertical]
if is_saved == 'Saved Card Only': df = df[df['is_saved_card'] == True]
elif is_saved == 'New Card Only': df = df[df['is_saved_card'] == False]

st.title("🛡️ Fraud Prevention Efficient Frontier")

if df.empty:
    st.warning("No data available for this combination of filters. Please adjust your selections.")
    st.stop()

# --- 3. BASELINE CALCULATIONS ---
total_gmv = df['amount_eur'].sum()
actual_fraud_gmv = df[df['is_actual_chargeback'] == 1]['amount_eur'].sum()
actual_good_gmv = df[df['is_actual_chargeback'] == 0]['amount_eur'].sum()

# Current Hardcoded State Calculations
curr_tp = df[(df['rule_name'] != 'No_Rule') & (df['order_final_status'] == 'CANCELLED') & (df['is_actual_chargeback'] == 1)]
curr_benefit = curr_tp['amount_eur'].sum() + (len(curr_tp) * cb_fee)

curr_fp = df[(df['rule_name'] != 'No_Rule') & (df['order_final_status'] == 'CANCELLED') & (df['is_actual_chargeback'] == 0)]
curr_margin_loss = curr_fp['amount_eur'].sum() * margin_pct
curr_friction = curr_margin_loss + (len(curr_fp[curr_fp['payment_switch'] == 0]) * ltv_cost)

# --- DYNAMIC KPIs ---
st.markdown("### Financial Impact")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Processed GMV", f"€{total_gmv:,.0f}")
col2.metric("Underlying Fraud Exposure", f"€{actual_fraud_gmv:,.0f}")

# --- 4. CALCULATE FRONTIER ---
def calculate_frontier(df_subset, margin, fee, ltv):
    thresholds = np.linspace(5, 95, 40)
    curve = []
    
    for t in thresholds:
        sim_blocked = df_subset['risk_score'] >= t
        tp_mask = sim_blocked & (df_subset['is_actual_chargeback'] == 1)
        benefit = df_subset.loc[tp_mask, 'amount_eur'].sum() + (tp_mask.sum() * fee)
        
        fp_mask = sim_blocked & (df_subset['is_actual_chargeback'] == 0)
        fp_df = df_subset[fp_mask]
        # Use underlying propensity to see if simulated block would have switched
        friction = (fp_df['amount_eur'].sum() * margin) + (len(fp_df[fp_df['switch_propensity'] == 0]) * ltv)
        
        curve.append({'Threshold': t, 'Benefit (€)': benefit, 'Friction (€)': friction, 'NFI': benefit - friction})
        
    df_c = pd.DataFrame(curve)
    opt_row = df_c.loc[df_c['NFI'].idxmax()]
    
    # Safely calculate actual equivalent
    df_c['Friction_Diff'] = abs(df_c['Friction (€)'] - curr_friction)
    actual_eq_row = df_c.loc[df_c['Friction_Diff'].idxmin()]
    
    return df_c, opt_row, actual_eq_row['Threshold']

df_curve, opt_point, actual_eq_thresh = calculate_frontier(df, margin_pct, cb_fee, ltv_cost)

col3.metric("Max Net Financial Impact", f"€{opt_point['NFI']:,.0f}")
col4.metric("Optimal Risk Threshold", f"{opt_point['Threshold']:.1f}")

st.markdown("### Current System Accuracy")
acc1, acc2, acc3, acc4 = st.columns(4)
capture_rate = (curr_tp['amount_eur'].sum() / actual_fraud_gmv) * 100 if actual_fraud_gmv > 0 else 0
fpr = (curr_fp['amount_eur'].sum() / actual_good_gmv) * 100 if actual_good_gmv > 0 else 0
overall_precision = (len(curr_tp) / (len(curr_tp) + len(curr_fp))) * 100 if (len(curr_tp) + len(curr_fp)) > 0 else 0

acc1.metric("Fraud Capture Rate", f"{capture_rate:.1f}%")
acc2.metric("False Positive Rate", f"{fpr:.2f}%", delta_color="inverse")
acc3.metric("System Precision", f"{overall_precision:.1f}%")
acc4.metric("Friction Ratio", f"1 : {(len(curr_fp)/len(curr_tp)):.1f}" if len(curr_tp) > 0 else "N/A")

# --- 5. PLOT EFFICIENT FRONTIER ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_curve['Friction (€)'], y=df_curve['Benefit (€)'], mode='lines', name='Risk Score Frontier', line=dict(color='royalblue', width=3)))
fig.add_trace(go.Scatter(x=[opt_point['Friction (€)']], y=[opt_point['Benefit (€)']], mode='markers', name='Optimal Setup (Peak NFI)', marker=dict(color='orange', size=14, symbol='star')))
fig.add_trace(go.Scatter(x=[curr_friction], y=[curr_benefit], mode='markers', name='Current Rules State', marker=dict(color='red', size=12, symbol='x')))

fig.update_layout(title='Efficient Frontier: Where we are vs. Where we could be', xaxis_title='Cost of Friction (Lost Margin + Churn LTV) €', yaxis_title='Fraud & Fees Prevented €', hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- 6. BENCHMARK TABLE ---
st.subheader("Market & Monthly Threshold Benchmarks")
benchmark_data = []
for (mkt, mth), group_df in df.groupby(['global_entity_id', 'Month']):
    if len(group_df) < 50: continue 
    
    # Calculate group current friction
    g_fp = group_df[(group_df['rule_name'] != 'No_Rule') & (group_df['order_final_status'] == 'CANCELLED') & (group_df['is_actual_chargeback'] == 0)]
    g_curr_friction = (g_fp['amount_eur'].sum() * margin_pct) + (len(g_fp[g_fp['payment_switch'] == 0]) * ltv_cost)
    
    # Build curve
    curve = []
    for t in np.linspace(5, 95, 40):
        sim_blocked = group_df['risk_score'] >= t
        tp_mask = sim_blocked & (group_df['is_actual_chargeback'] == 1)
        ben = group_df.loc[tp_mask, 'amount_eur'].sum() + (tp_mask.sum() * cb_fee)
        
        fp_mask = sim_blocked & (group_df['is_actual_chargeback'] == 0)
        fp_df = group_df[fp_mask]
        fric = (fp_df['amount_eur'].sum() * margin_pct) + (len(fp_df[fp_df['switch_propensity'] == 0]) * ltv_cost)
        curve.append({'Threshold': t, 'NFI': ben - fric, 'Friction': fric})
        
    df_c = pd.DataFrame(curve)
    opt_t = df_c.loc[df_c['NFI'].idxmax()]['Threshold']
    act_t = df_c.loc[abs(df_c['Friction'] - g_curr_friction).idxmin()]['Threshold']
    
    diff = opt_t - act_t
    status = "Too Stiff (Over-blocking)" if diff > 5 else "Too Loose (Under-blocking)" if diff < -5 else "Target Met"
    benchmark_data.append({'Market': mkt, 'Month': mth, 'Optimal Threshold': round(opt_t, 1), 'Equivalent Actual Threshold': round(act_t, 1), 'Difference': round(diff, 1), 'Status': status})

if benchmark_data:
    df_bench = pd.DataFrame(benchmark_data).sort_values(by=['Market', 'Month'])
    st.dataframe(df_bench.style.map(lambda v: f"color: {'red' if 'Over' in v else 'orange' if 'Under' in v else 'green'}", subset=['Status']), use_container_width=True, hide_index=True)

# --- 7. RULE LEVEL ATTRIBUTION ---
st.subheader("Granular Rule Performance & Accuracy")
rule_stats = []
active_rules = df[df['rule_name'] != 'No_Rule']['rule_name'].unique()

for rule in active_rules:
    r_df = df[df['rule_name'] == rule]
    if r_df.empty: continue
    
    tp_df = r_df[(r_df['is_actual_chargeback'] == 1) & (r_df['order_final_status'] == 'CANCELLED')]
    ben = tp_df['amount_eur'].sum() + (len(tp_df) * cb_fee)
    
    fp_df = r_df[(r_df['is_actual_chargeback'] == 0) & (r_df['order_final_status'] == 'CANCELLED')]
    fric = (fp_df['amount_eur'].sum() * margin_pct) + (len(fp_df[fp_df['payment_switch'] == 0]) * ltv_cost)
    
    rule_stats.append({
        'Rule Name': rule,
        'Action': r_df['rule_action'].iloc[0],
        'Capture Rate (%)': (tp_df['amount_eur'].sum() / actual_fraud_gmv * 100) if actual_fraud_gmv > 0 else 0,
        'Precision (%)': (len(tp_df) / (len(tp_df) + len(fp_df)) * 100) if (len(tp_df) + len(fp_df)) > 0 else 0,
        'Friction Ratio': f"1 : {(len(fp_df)/len(tp_df)):.1f}" if len(tp_df) > 0 else "∞",
        'TPs (Caught Fraud)': len(tp_df),
        'FPs (Lost Users)': len(fp_df),
        'Net Impact (€)': ben - fric
    })

if rule_stats:
    st.dataframe(pd.DataFrame(rule_stats).sort_values('Net Impact (€)', ascending=False).style.format({'Net Impact (€)': '€{:,.0f}', 'Capture Rate (%)': '{:.1f}%', 'Precision (%)': '{:.1f}%'}), use_container_width=True)
