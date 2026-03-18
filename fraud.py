import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Optimization Engine", layout="wide")

# --- 1. UNIFIED DATA GENERATOR & LOADER ---
@st.cache_data
def get_or_create_data():
    np.random.seed(777) 
    num_records = 30000
    markets = ['FO_NO', 'FO_SE', 'FO_FI', 'FO_DK', 'FO_EE']
    entity_col = np.random.choice(markets, num_records, p=[0.25, 0.25, 0.20, 0.15, 0.15])
    dates = pd.date_range(start='2026-01-01', periods=num_records, freq='5T')
    amount_eur = np.random.lognormal(mean=3.2, sigma=0.4, size=num_records).round(2) 

    vendor_vertical = np.random.choice(['restaurants', 'supermarket', 'darkstores'], num_records, p=[0.6, 0.3, 0.1])
    customer_source_type = np.random.choice(['New', 'Existing'], num_records, p=[0.25, 0.75])
    is_saved_card = np.where(customer_source_type == 'Existing', np.random.choice([True, False], num_records, p=[0.8, 0.2]), False)
    is_3ds_eligible = np.random.choice([True, False], num_records, p=[0.9, 0.1])

    is_actual_chargeback = np.random.binomial(1, 0.04, num_records)

    risk_score = np.zeros(num_records)
    for i in range(num_records):
        if is_actual_chargeback[i] == 1:
            risk_score[i] = np.random.normal(70, 20) 
        else:
            risk_score[i] = np.random.normal(25, 15) 
            
    risk_score = np.where(entity_col == 'FO_EE', risk_score + 8, risk_score)
    risk_score = np.where(entity_col == 'FO_NO', risk_score - 8, risk_score)
    risk_score = np.clip(risk_score, 0, 99).round(2)

    rule_triggered = np.array(['No_Rule'] * num_records, dtype=object)
    for i in range(num_records):
        mkt, score = entity_col[i], risk_score[i]
        if mkt == 'FO_NO' and score > 45: rule_triggered[i] = 'velocity_high_risk_block'
        elif mkt == 'FO_SE' and score > 55: rule_triggered[i] = 'card_limit_value_3ds'
        elif mkt == 'FO_EE' and score > 65: rule_triggered[i] = 'foreign_card_review'
        elif mkt == 'FO_FI' and score > 50: rule_triggered[i] = 'card_account_age_7d_3ds'
        elif mkt == 'FO_DK' and score > 55: rule_triggered[i] = 'card_limit_value_3ds'

    rule_action = np.where(pd.Series(rule_triggered).str.contains('block'), 'hard_block', 
                  np.where(pd.Series(rule_triggered) == 'No_Rule', 'none', 'review'))

    switch_propensity = np.random.binomial(1, 0.35, num_records)
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
                    drop_rate = 0.6 if not is_3ds_eligible[i] else 0.2
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

# --- 2. FRONTIER CALCULATION ENGINE ---
def calculate_frontier(df_subset, margin, fee, ltv, target_friction):
    thresholds = np.linspace(1, 99, 200)
    curve = []
    non_rule_df = df_subset[df_subset['rule_name'] != 'No_Rule']
    primary_action = non_rule_df['rule_action'].mode()[0] if not non_rule_df.empty else 'hard_block'

    for t in thresholds:
        sim_blocked = df_subset['risk_score'] >= t
        tp_mask = sim_blocked & (df_subset['is_actual_chargeback'] == 1)
        benefit = df_subset.loc[tp_mask, 'amount_eur'].sum() + (tp_mask.sum() * fee)
        
        fp_mask = sim_blocked & (df_subset['is_actual_chargeback'] == 0)
        fp_df = df_subset[fp_mask]
        
        drop_rate = 0.20 if primary_action == 'review' else 1.0
        friction = (fp_df['amount_eur'].sum() * drop_rate * margin) + (len(fp_df) * drop_rate * 0.65 * ltv)
        
        missed_mask = (~sim_blocked) & (df_subset['is_actual_chargeback'] == 1)
        missed_loss = df_subset.loc[missed_mask, 'amount_eur'].sum() + (missed_mask.sum() * fee)
        
        nfi = benefit - friction - missed_loss
        curve.append({'Threshold': t, 'Benefit (€)': benefit, 'Friction (€)': friction, 'NFI': nfi})
        
    df_c = pd.DataFrame(curve)
    opt_row = df_c.loc[df_c['NFI'].idxmax()]
    df_c['Friction_Diff'] = abs(df_c['Friction (€)'] - target_friction)
    actual_eq_row = df_c.loc[df_c['Friction_Diff'].idxmin()]
    
    return df_c, opt_row, actual_eq_row['Threshold']

# --- 3. UI & FILTERS ---
df_raw = get_or_create_data()

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

# Filter Application
df = df_raw.copy()
if market != 'All Markets': df = df[df['global_entity_id'] == market]
if month != 'All Months': df = df[df['Month'] == month]
if vertical != 'All Verticals': df = df[df['vendor_vertical'] == vertical]
if is_saved == 'Saved Card Only': df = df[df['is_saved_card'] == True]
elif is_saved == 'New Card Only': df = df[df['is_saved_card'] == False]

st.title("🛡️ Fraud Prevention Efficient Frontier")

# --- 4. ACTUALS CALCULATIONS ---
total_gmv = df['amount_eur'].sum()
actual_fraud_exposure = df[df['is_actual_chargeback'] == 1]['amount_eur'].sum()
curr_tp = df[(df['rule_name'] != 'No_Rule') & (df['order_final_status'] == 'CANCELLED') & (df['is_actual_chargeback'] == 1)]
curr_benefit = curr_tp['amount_eur'].sum() + (len(curr_tp) * cb_fee)
curr_fp = df[(df['rule_name'] != 'No_Rule') & (df['order_final_status'] == 'CANCELLED') & (df['is_actual_chargeback'] == 0)]
curr_friction = (curr_fp['amount_eur'].sum() * margin_pct) + (len(curr_fp[curr_fp['payment_switch'] == 0]) * ltv_cost)
fraud_missed = df[(df['is_actual_chargeback'] == 1) & (df['order_final_status'] == 'DELIVERED')]
curr_fraud_loss = fraud_missed['amount_eur'].sum() + (len(fraud_missed) * cb_fee)
actual_nfi = curr_benefit - curr_friction - curr_fraud_loss

# --- 5. SECTION 1: FINANCIAL IMPACT ---
st.markdown("### 1. Actual Financial Impact (Current Rules State)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Net Financial Impact", f"€{actual_nfi:,.0f}")
c2.metric("Fraud Prevented (Benefit)", f"€{curr_benefit:,.0f}")
c3.metric("Cost of False Positives", f"-€{curr_friction:,.0f}")
c4.metric("Cost of Fraud (Missed)", f"-€{curr_fraud_loss:,.0f}")

cr1, cr2, cr3, cr4 = st.columns(4)
cr1.metric("NFI Rate", f"{(actual_nfi/total_gmv)*100:.2f}%")
cr2.metric("Benefit Rate", f"{(curr_benefit/total_gmv)*100:.2f}%")
cr3.metric("FP Rate (GMV)", f"{(curr_friction/total_gmv)*100:.2f}%", delta_color="inverse")
cr4.metric("Fraud Loss Rate", f"{(curr_fraud_loss/total_gmv)*100:.2f}%", delta_color="inverse")
st.divider()

# --- 6. SECTION 2: OPTIMIZATION ---
if market == 'All Markets':
    sum_max_nfi = 0
    df_curve, opt_point, actual_eq_thresh = calculate_frontier(df, margin_pct, cb_fee, ltv_cost, curr_friction)
    for m in df['global_entity_id'].unique():
        m_df = df[df['global_entity_id'] == m]
        m_fp_curr = m_df[(m_df['rule_name'] != 'No_Rule') & (m_df['order_final_status'] == 'CANCELLED') & (m_df['is_actual_chargeback'] == 0)]
        m_fric_curr = (m_fp_curr['amount_eur'].sum() * margin_pct) + (len(m_fp_curr[m_fp_curr['payment_switch'] == 0]) * ltv_cost)
        _, m_opt, _ = calculate_frontier(m_df, margin_pct, cb_fee, ltv_cost, m_fric_curr)
        sum_max_nfi += m_opt['NFI']
    max_nfi_display, opt_thresh_display = sum_max_nfi, "Market-Specific"
else:
    df_curve, opt_point, actual_eq_thresh = calculate_frontier(df, margin_pct, cb_fee, ltv_cost, curr_friction)
    max_nfi_display, opt_thresh_display = opt_point['NFI'], f"{opt_point['Threshold']:.1f}"

st.markdown("### 2. Optimal vs. Actual Benchmark")
b1, b2, b3, b4 = st.columns(4)
b1.metric("Total GMV", f"€{total_gmv:,.0f}")
exposure_delta = actual_fraud_exposure - (curr_benefit - (len(curr_tp) * cb_fee))
b2.metric("Underlying Fraud Exposure", f"€{actual_fraud_exposure:,.0f}", f"-€{exposure_delta:,.0f} caught")
nfi_gap = max_nfi_display - actual_nfi
b3.metric("Max Net Financial Impact", f"€{max_nfi_display:,.0f}", f"{nfi_gap:+,.0f} vs Actual", delta_color="inverse")
b4.metric("Optimal Risk Threshold", opt_thresh_display, f"vs Actual eq. {actual_eq_thresh:.1f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_curve['Friction (€)'], y=df_curve['Benefit (€)'], mode='lines', name='Risk Score Frontier', line=dict(color='royalblue', width=3)))
fig.add_trace(go.Scatter(x=[opt_point['Friction (€)']], y=[opt_point['Benefit (€)']], mode='markers', name='Optimal Point', marker=dict(color='orange', size=14, symbol='star')))
fig.add_trace(go.Scatter(x=[curr_friction], y=[curr_benefit], mode='markers', name='Current Rules', marker=dict(color='red', size=12, symbol='x')))
fig.update_layout(title='Efficient Frontier', xaxis_title='Friction Cost €', yaxis_title='Benefit €', hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
st.divider()

# --- 7. SECTION 3: DEEP DIVES (WITH FIXED COLORS) ---
st.markdown("### 3. Friction Deep-Dive")
fd1, fd2, fd3 = st.columns(3)

if not curr_fp.empty:
    recovered = int(curr_fp['payment_switch'].sum())
    fig_f = go.Figure(go.Funnel(
        y=["Blocked", "Lost", "Recovered"], 
        x=[len(curr_fp), len(curr_fp)-recovered, recovered],
        marker={"color": ["#636EFA", "#EF553B", "#00CC96"]}
    ))
    fd1.plotly_chart(fig_f, use_container_width=True)
    
    fp_new, fp_old = curr_fp[curr_fp['customer_source_type'] == 'New'], curr_fp[curr_fp['customer_source_type'] == 'Existing']
    f_new = (fp_new['amount_eur'].sum() * margin_pct) + (len(fp_new[fp_new['payment_switch'] == 0]) * ltv_cost)
    f_old = (fp_old['amount_eur'].sum() * margin_pct) + (len(fp_old[fp_old['payment_switch'] == 0]) * ltv_cost)
    fd2.plotly_chart(go.Figure(data=[go.Pie(labels=['New', 'Existing'], values=[f_new, f_old], hole=.4, marker_colors=["#AB63FA", "#FFA15A"])]), use_container_width=True)
    
    act_counts = curr_fp['rule_action'].value_counts()
    fd3.plotly_chart(go.Figure(data=[go.Pie(labels=['3DS Drop-off' if x=='review' else 'Hard Decline' for x in act_counts.index], values=act_counts.values, hole=.4, marker_colors=["#19D3F3", "#FF6692"])]), use_container_width=True)
st.divider()

# --- 8. TABLES ---
st.subheader("Market Benchmarks")
bench_list = []
for (mkt, mth), g_df in df.groupby(['global_entity_id', 'Month']):
    if len(g_df) < 50: continue
    g_fp = g_df[(g_df['rule_name'] != 'No_Rule') & (g_df['order_final_status'] == 'CANCELLED') & (g_df['is_actual_chargeback'] == 0)]
    g_fric = (g_fp['amount_eur'].sum() * margin_pct) + (len(g_fp[g_fp['payment_switch'] == 0]) * ltv_cost)
    _, g_opt, g_act_t = calculate_frontier(g_df, margin_pct, cb_fee, ltv_cost, g_fric)
    diff = g_opt['Threshold'] - g_act_t
    status = "Too Stiff" if diff > 5 else "Too Loose" if diff < -5 else "Target Met"
    bench_list.append({'Market': mkt, 'Month': mth, 'Opt Thresh': round(g_opt['Threshold'], 1), 'Act Eq Thresh': round(g_act_t, 1), 'Status': status})

if bench_list:
    df_b = pd.DataFrame(bench_list)
    def style_status(v):
        color = '#EF553B' if v == "Too Stiff" else '#FFA15A' if v == "Too Loose" else '#00CC96'
        return f'color: {color}; font-weight: bold'
    st.dataframe(df_b.style.applymap(style_status, subset=['Status']), use_container_width=True, hide_index=True)

st.subheader("Rule Level Breakdown")
rule_list = []
for rule in df[df['rule_name'] != 'No_Rule']['rule_name'].unique():
    r_df = df[df['rule_name'] == rule]
    rtp = r_df[(r_df['is_actual_chargeback'] == 1) & (r_df['order_final_status'] == 'CANCELLED')]
    rfp = r_df[(r_df['is_actual_chargeback'] == 0) & (r_df['order_final_status'] == 'CANCELLED')]
    ben = rtp['amount_eur'].sum() + (len(rtp) * cb_fee)
    fric = (rfp['amount_eur'].sum() * margin_pct) + (len(rfp[rfp['payment_switch'] == 0]) * ltv_cost)
    rule_list.append({'Rule': rule, 'Benefit': ben, 'Friction': fric, 'Net Impact': ben-fric, 'Precision': len(rtp)/(len(rtp)+len(rfp))*100 if (len(rtp)+len(rfp))>0 else 0})

if rule_list:
    df_r = pd.DataFrame(rule_list).sort_values('Net Impact', ascending=False)
    st.dataframe(df_r.style.format({'Benefit': '€{:,.0f}', 'Friction': '€{:,.0f}', 'Net Impact': '€{:,.0f}', 'Precision': '{:.1f}%'}), use_container_width=True, hide_index=True)

# --- 9. ACTION INSIGHT BOX ---
st.divider()
st.subheader("💡 Optimization Recommendation")
if rule_list:
    worst_rule = min(rule_list, key=lambda x: x['Net Impact'])
    if worst_rule['Net Impact'] < 0:
        st.error(f"**High Priority:** Rule `{worst_rule['Rule']}` is currently destroying value. Its Friction (Lost LTV) is €{worst_rule['Friction']:,.0f}, which outweighs the fraud it catches. Consider increasing its risk threshold immediately.")
    else:
        st.success("All active rules are currently contributing positively to Net Financial Impact. Focus on micro-tuning thresholds to move closer to the Orange Star.")
