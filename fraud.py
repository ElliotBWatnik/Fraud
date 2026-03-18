import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Optimization Engine", layout="wide")

# --- 1. UNIFIED DATA GENERATOR & LOADER (HIGHLY POSITIVE SKEW) ---
@st.cache_data
def get_or_create_data():
    np.random.seed(999) 
    num_records = 30000
    
    markets = ['FO_NO', 'FO_SE', 'FO_FI', 'FO_DK', 'FO_EE']
    entity_col = np.random.choice(markets, num_records, p=[0.25, 0.25, 0.20, 0.15, 0.15])
    dates = pd.date_range(start='2026-01-01', periods=num_records, freq='5T')
    amount_eur = np.random.lognormal(mean=3.2, sigma=0.4, size=num_records).round(2) 

    vendor_vertical = np.random.choice(['restaurants', 'supermarket', 'darkstores'], num_records, p=[0.6, 0.3, 0.1])
    customer_source_type = np.random.choice(['New', 'Existing'], num_records, p=[0.25, 0.75])
    is_saved_card = np.where(customer_source_type == 'Existing', np.random.choice([True, False], num_records, p=[0.8, 0.2]), False)
    issuer_country_group = np.random.choice(['Local', 'International'], num_records, p=[0.85, 0.15])
    is_3ds_eligible = np.random.choice([True, False], num_records, p=[0.9, 0.1])

    base_fraud_prob = 0.04
    is_actual_chargeback = np.random.binomial(1, base_fraud_prob, num_records)

    risk_score = np.zeros(num_records)
    for i in range(num_records):
        if is_actual_chargeback[i] == 1:
            risk_score[i] = np.random.normal(85, 10)
        else:
            risk_score[i] = np.random.normal(15, 10)
            
    risk_score = np.where(entity_col == 'FO_EE', risk_score + 5, risk_score)
    risk_score = np.where(entity_col == 'FO_NO', risk_score - 5, risk_score)
    risk_score = np.clip(risk_score, 0, 99).round(2)

    rule_triggered = np.array(['No_Rule'] * num_records, dtype=object)
    for i in range(num_records):
        mkt = entity_col[i]
        score = risk_score[i]
        if mkt == 'FO_NO':
            if score > 50: rule_triggered[i] = 'velocity_high_risk_block'
        elif mkt == 'FO_SE': 
            if score > 45: rule_triggered[i] = 'card_limit_value_3ds'
        elif mkt == 'FO_EE': 
            if score > 55: rule_triggered[i] = 'foreign_card_review'
        elif mkt == 'FO_FI':
            if score > 48: rule_triggered[i] = 'card_account_age_7d_3ds'
        else: 
            if score > 45: rule_triggered[i] = 'card_limit_value_3ds'

    rule_action = np.where(pd.Series(rule_triggered).str.contains('block'), 'hard_block', 
                  np.where(pd.Series(rule_triggered) == 'No_Rule', 'none', 'review'))

    switch_propensity = np.random.binomial(1, 0.40, num_records)
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
                    drop_rate = 0.5 if not is_3ds_eligible[i] else 0.15
                    if np.random.rand() < drop_rate:
                        order_final_status[i] = 'CANCELLED'
                        payment_switch[i] = switch_propensity[i]

    df = pd.DataFrame({
        'order_date': dates,
        'Month': dates.strftime('%Y-%m'),
        'global_entity_id': entity_col,
        'amount_eur': amount_eur,
        'vendor_vertical': vendor_vertical,
        'customer_source_type': customer_source_type,
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

market = st.sidebar.selectbox("Market Entity", options=['All Markets'] + sorted(list(df_raw['global_entity_id'].unique())))
month = st.sidebar.selectbox("Month", options=['All Months'] + sorted(list(df_raw['Month'].unique())))
vertical = st.sidebar.selectbox("Vendor Vertical", options=['All Verticals'] + sorted(list(df_raw['vendor_vertical'].unique())))
is_saved = st.sidebar.selectbox("Card Status", options=['All Cards', 'Saved Card Only', 'New Card Only'])

st.sidebar.header("2. Economic Assumptions")
margin_pct = st.sidebar.slider("Net Profit Margin (%)", 1, 30, 10) / 100.0
cb_fee = st.sidebar.number_input("Chargeback Fee (€)", value=15.0)
ltv_cost = st.sidebar.number_input("Blended LTV Cost per Churn (€)", value=50.0)

# --- 3. APPLY FILTERS ---
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

# --- 4. BASELINE & ACTUALS CALCULATIONS ---
total_gmv = df['amount_eur'].sum()
actual_fraud_exposure = df[df['is_actual_chargeback'] == 1]['amount_eur'].sum()

# Current Hardcoded State Calculations
curr_tp = df[(df['rule_name'] != 'No_Rule') & (df['order_final_status'] == 'CANCELLED') & (df['is_actual_chargeback'] == 1)]
curr_benefit = curr_tp['amount_eur'].sum() + (len(curr_tp) * cb_fee)

curr_fp = df[(df['rule_name'] != 'No_Rule') & (df['order_final_status'] == 'CANCELLED') & (df['is_actual_chargeback'] == 0)]
curr_margin_loss = curr_fp['amount_eur'].sum() * margin_pct
curr_friction = curr_margin_loss + (len(curr_fp[curr_fp['payment_switch'] == 0]) * ltv_cost)

fraud_missed = df[(df['is_actual_chargeback'] == 1) & (df['order_final_status'] == 'DELIVERED')]
curr_fraud_loss = fraud_missed['amount_eur'].sum() + (len(fraud_missed) * cb_fee)

actual_nfi = curr_benefit - curr_friction - curr_fraud_loss

# --- 5. ACTUAL FINANCIAL IMPACT & RATES ---
st.markdown("### 1. Actual Financial Impact (Current Rules State)")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Net Financial Impact", f"€{actual_nfi:,.0f}")
col2.metric("Fraud Prevented (Benefit)", f"€{curr_benefit:,.0f}")
col3.metric("Cost of False Positives", f"-€{curr_friction:,.0f}")
col4.metric("Cost of Fraud (Missed)", f"-€{curr_fraud_loss:,.0f}")

col1_r, col2_r, col3_r, col4_r = st.columns(4)
nfi_rate = (actual_nfi / total_gmv) * 100 if total_gmv > 0 else 0
benefit_rate = (curr_benefit / total_gmv) * 100 if total_gmv > 0 else 0
friction_rate = (curr_friction / total_gmv) * 100 if total_gmv > 0 else 0
fraud_loss_rate = (curr_fraud_loss / total_gmv) * 100 if total_gmv > 0 else 0

col1_r.metric("NFI Rate (% of GMV)", f"{nfi_rate:.2f}%")
col2_r.metric("Fraud Prevented Rate", f"{benefit_rate:.2f}%")
col3_r.metric("False Positive Rate", f"{friction_rate:.2f}%", delta_color="inverse")
col4_r.metric("Fraud Loss Rate", f"{fraud_loss_rate:.2f}%", delta_color="inverse")

st.divider()

# --- 6. CALCULATE ML FRONTIER ---
def calculate_frontier(df_subset, margin, fee, ltv, target_friction):
    thresholds = np.linspace(5, 95, 40)
    curve = []
    
    for t in thresholds:
        sim_blocked = df_subset['risk_score'] >= t
        
        tp_mask = sim_blocked & (df_subset['is_actual_chargeback'] == 1)
        benefit = df_subset.loc[tp_mask, 'amount_eur'].sum() + (tp_mask.sum() * fee)
        
        fp_mask = sim_blocked & (df_subset['is_actual_chargeback'] == 0)
        fp_df = df_subset[fp_mask]
        friction = (fp_df['amount_eur'].sum() * margin) + (len(fp_df[fp_df['switch_propensity'] == 0]) * ltv)
        
        missed_mask = (~sim_blocked) & (df_subset['is_actual_chargeback'] == 1)
        missed_loss = df_subset.loc[missed_mask, 'amount_eur'].sum() + (missed_mask.sum() * fee)
        
        nfi = benefit - friction - missed_loss
        curve.append({'Threshold': t, 'Benefit (€)': benefit, 'Friction (€)': friction, 'NFI': nfi})
        
    df_c = pd.DataFrame(curve)
    opt_row = df_c.loc[df_c['NFI'].idxmax()]
    
    df_c['Friction_Diff'] = abs(df_c['Friction (€)'] - target_friction)
    actual_eq_row = df_c.loc[df_c['Friction_Diff'].idxmin()]
    
    return df_c, opt_row, actual_eq_row['Threshold']

df_curve, opt_point, actual_eq_thresh = calculate_frontier(df, margin_pct, cb_fee, ltv_cost, curr_friction)

# --- 7. OPTIMAL VS ACTUAL BENCHMARK ---
st.markdown("### 2. Optimal vs. Actual Benchmark")
b1, b2, b3, b4 = st.columns(4)

b1.metric("Total Processed GMV", f"€{total_gmv:,.0f}")
exposure_delta = actual_fraud_exposure - curr_fraud_loss
b2.metric("Underlying Fraud Exposure", f"€{actual_fraud_exposure:,.0f}", f"-€{exposure_delta:,.0f} caught", delta_color="normal")
nfi_delta = opt_point['NFI'] - actual_nfi
b3.metric("Max Net Financial Impact", f"€{opt_point['NFI']:,.0f}", f"{nfi_delta:+,.0f} vs Actual", delta_color="normal" if nfi_delta > 0 else "off")
thresh_delta = opt_point['Threshold'] - actual_eq_thresh
b4.metric("Optimal Risk Threshold", f"{opt_point['Threshold']:.1f}", f"{thresh_delta:+.1f} vs Actual eq.", delta_color="off")

# --- 8. PLOT EFFICIENT FRONTIER ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_curve['Friction (€)'], y=df_curve['Benefit (€)'], mode='lines', name='Risk Score Frontier', line=dict(color='royalblue', width=3)))
fig.add_trace(go.Scatter(x=[opt_point['Friction (€)']], y=[opt_point['Benefit (€)']], mode='markers', name='Optimal Setup (Peak NFI)', marker=dict(color='orange', size=14, symbol='star')))
fig.add_trace(go.Scatter(x=[curr_friction], y=[curr_benefit], mode='markers', name='Current Rules State', marker=dict(color='red', size=12, symbol='x')))
fig.update_layout(title='Efficient Frontier: Where we are vs. Where we could be', xaxis_title='Cost of Friction (Lost Margin + Churn LTV) €', yaxis_title='Fraud & Fees Prevented €', hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- 9. FRICTION DEEP-DIVE VISUALS ---
st.markdown("### 3. Friction Deep-Dive: Where are we losing money?")
fig_col1, fig_col2, fig_col3 = st.columns(3)

if curr_fp.empty:
    st.info("No False Positives generated by the current filters. Your friction is €0!")
else:
    # Funnel
    total_fp_count = len(curr_fp)
    recovered_count = int(curr_fp['payment_switch'].sum())
    churned_count = total_fp_count - recovered_count

    fig_funnel = go.Figure(go.Funnel(
        y=["Total Blocked (Good)", "Lost / Churned", "Recovered (Switched)"],
        x=[total_fp_count, churned_count, recovered_count],
        textinfo="value+percent initial",
        marker={"color": ["#636EFA", "#EF553B", "#00CC96"]}
    ))
    fig_funnel.update_layout(title="Payment Switch Recovery", margin=dict(t=40, b=20, l=0, r=0))
    fig_col1.plotly_chart(fig_funnel, use_container_width=True)

    # Donut
    fp_new = curr_fp[curr_fp['customer_source_type'] == 'New']
    fp_exist = curr_fp[curr_fp['customer_source_type'] == 'Existing']
    
    fric_new = (fp_new['amount_eur'].sum() * margin_pct) + (len(fp_new[fp_new['payment_switch'] == 0]) * ltv_cost)
    fric_exist = (fp_exist['amount_eur'].sum() * margin_pct) + (len(fp_exist[fp_exist['payment_switch'] == 0]) * ltv_cost)

    fig_pie1 = go.Figure(data=[go.Pie(labels=['New Users', 'Existing Users'], values=[fric_new, fric_exist], hole=.4, marker_colors=["#AB63FA", "#FFA15A"])])
    fig_pie1.update_layout(title="Friction Cost (€) by User Type", margin=dict(t=40, b=20, l=0, r=0))
    fig_col2.plotly_chart(fig_pie1, use_container_width=True)

    # Pie
    action_counts = curr_fp['rule_action'].value_counts().reset_index()
    action_counts.columns = ['Action', 'Count']
    action_counts['Action'] = action_counts['Action'].map({'review': '3DS Drop-off', 'hard_block': 'Hard Decline'})

    fig_pie2 = go.Figure(data=[go.Pie(labels=action_counts['Action'], values=action_counts['Count'], hole=.4, marker_colors=["#19D3F3", "#FF6692"])])
    fig_pie2.update_layout(title="False Positives by Action Type", margin=dict(t=40, b=20, l=0, r=0))
    fig_col3.plotly_chart(fig_pie2, use_container_width=True)

st.divider()

# --- 10. MONTHLY BENCHMARK TABLE ---
st.subheader("Market & Monthly Threshold Benchmarks")
benchmark_data = []
for (mkt, mth), group_df in df.groupby(['global_entity_id', 'Month']):
    if len(group_df) < 50: continue 
    
    g_fp = group_df[(group_df['rule_name'] != 'No_Rule') & (group_df['order_final_status'] == 'CANCELLED') & (group_df['is_actual_chargeback'] == 0)]
    g_curr_friction = (g_fp['amount_eur'].sum() * margin_pct) + (len(g_fp[g_fp['payment_switch'] == 0]) * ltv_cost)
    
    _, opt_row, act_thresh = calculate_frontier(group_df, margin_pct, cb_fee, ltv_cost, g_curr_friction)
    
    diff = opt_row['Threshold'] - act_thresh
    status = "Too Stiff (Over-blocking)" if diff > 5 else "Too Loose (Under-blocking)" if diff < -5 else "Target Met"
    benchmark_data.append({'Market': mkt, 'Month': mth, 'Optimal Threshold': round(opt_row['Threshold'], 1), 'Equivalent Actual Threshold': round(act_thresh, 1), 'Difference': round(diff, 1), 'Status': status})

if benchmark_data:
    df_bench = pd.DataFrame(benchmark_data).sort_values(by=['Market', 'Month'])
    st.dataframe(df_bench.style.map(lambda v: f"color: {'red' if 'Over' in v else 'orange' if 'Under' in v else 'green'}", subset=['Status']), use_container_width=True, hide_index=True)

# --- 11. RULE LEVEL ATTRIBUTION ---
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
        'Capture Rate (%)': (tp_df['amount_eur'].sum() / actual_fraud_exposure * 100) if actual_fraud_exposure > 0 else 0,
        'Precision (%)': (len(tp_df) / (len(tp_df) + len(fp_df)) * 100) if (len(tp_df) + len(fp_df)) > 0 else 0,
        'Friction Ratio': f"1 : {(len(fp_df)/len(tp_df)):.1f}" if len(tp_df) > 0 else "∞",
        'TPs (Caught Fraud)': len(tp_df),
        'FPs (Lost Users)': len(fp_df),
        'Net Impact (€)': ben - fric
    })

if rule_stats:
    st.dataframe(pd.DataFrame(rule_stats).sort_values('Net Impact (€)', ascending=False).style.format({'Net Impact (€)': '€{:,.0f}', 'Capture Rate (%)': '{:.1f}%', 'Precision (%)': '{:.1f}%'}), use_container_width=True)
