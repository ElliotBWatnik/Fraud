import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Optimization Engine", layout="wide")

# --- DATA LOADING & PREP ---
@st.cache_data
def load_data():
    df = pd.read_csv('fraud_data.csv')
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['Month'] = df['order_date'].dt.strftime('%Y-%m')
    return df

df_raw = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("1. Global Filters")
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

# --- GLOBAL VARIABLES FOR BASELINE ---
total_gmv = df['amount_eur'].sum()
actual_fraud_gmv = df[df['is_actual_chargeback'] == 1]['amount_eur'].sum()
actual_good_gmv = df[df['is_actual_chargeback'] == 0]['amount_eur'].sum()

# Current Hardcoded State Calculations
curr_tp = df[(df['rule_name'] != 'None') & (df['order_final_status'] == 'CANCELLED') & (df['is_actual_chargeback'] == 1)]
curr_benefit = curr_tp['amount_eur'].sum() + (len(curr_tp) * cb_fee)

curr_fp = df[(df['rule_name'] != 'None') & (df['order_final_status'] == 'CANCELLED') & (df['is_actual_chargeback'] == 0)]
curr_margin_loss = curr_fp['amount_eur'].sum() * margin_pct
curr_friction = curr_margin_loss + (len(curr_fp[curr_fp['payment_switch'] == 0]) * ltv_cost)

# --- DYNAMIC KPIs (FINANCIAL) ---
st.markdown("### Financial Impact")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Processed GMV", f"€{total_gmv:,.0f}")
col2.metric("Underlying Fraud Exposure", f"€{actual_fraud_gmv:,.0f}")

# --- HELPER FUNCTION: CALCULATE FRONTIER & ACTUALS ---
def calculate_frontier(df_subset, margin, fee, ltv):
    thresholds = np.linspace(5, 95, 40)
    curve = []
    
    for t in thresholds:
        sim_blocked = df_subset['risk_score'] >= t
        tp_mask = sim_blocked & (df_subset['is_actual_chargeback'] == 1)
        benefit = df_subset.loc[tp_mask, 'amount_eur'].sum() + (tp_mask.sum() * fee)
        
        fp_mask = sim_blocked & (df_subset['is_actual_chargeback'] == 0)
        fp_df = df_subset[fp_mask]
        friction = (fp_df['amount_eur'].sum() * margin) + (len(fp_df[fp_df['payment_switch'] == 0]) * ltv)
        
        curve.append({'Threshold': t, 'Benefit (€)': benefit, 'Friction (€)': friction, 'NFI': benefit - friction})
        
    df_c = pd.DataFrame(curve)
    opt_row = df_c.loc[df_c['NFI'].idxmax()]
    
    df_c['Friction_Diff'] = abs(df_c['Friction (€)'] - curr_friction)
    actual_eq_row = df_c.loc[df_c['Friction_Diff'].idxmin()]
    
    return df_c, opt_row, actual_eq_row['Threshold']

df_curve, opt_point, actual_eq_thresh = calculate_frontier(df, margin_pct, cb_fee, ltv_cost)

col3.metric("Max Net Financial Impact", f"€{opt_point['NFI']:,.0f}")
col4.metric("Optimal Risk Threshold", f"{opt_point['Threshold']:.1f}")

# --- DYNAMIC KPIs (ACCURACY & CAPTURE RATES) ---
st.markdown("### Current System Accuracy")
acc1, acc2, acc3, acc4 = st.columns(4)

fraud_caught_gmv = curr_tp['amount_eur'].sum()
capture_rate = (fraud_caught_gmv / actual_fraud_gmv) * 100 if actual_fraud_gmv > 0 else 0

good_blocked_gmv = curr_fp['amount_eur'].sum()
fpr = (good_blocked_gmv / actual_good_gmv) * 100 if actual_good_gmv > 0 else 0

overall_precision = (len(curr_tp) / (len(curr_tp) + len(curr_fp))) * 100 if (len(curr_tp) + len(curr_fp)) > 0 else 0

acc1.metric("Fraud Capture Rate (Recall)", f"{capture_rate:.1f}%", help="What % of total fraud GMV did our rules catch?")
acc2.metric("False Positive Rate (GMV)", f"{fpr:.2f}%", help="What % of good user GMV was wrongly blocked?", delta_color="inverse")
acc3.metric("System Precision", f"{overall_precision:.1f}%", help="When a rule triggers, what is the probability it is actual fraud?")
acc4.metric("Friction Ratio", f"1 : {(len(curr_fp)/len(curr_tp)):.1f}" if len(curr_tp) > 0 else "N/A", help="For every 1 fraudster caught, how many good users are blocked?")

# --- PLOT EFFICIENT FRONTIER ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_curve['Friction (€)'], y=df_curve['Benefit (€)'], mode='lines', name='Risk Score Frontier', line=dict(color='royalblue', width=3)))
fig.add_trace(go.Scatter(x=[opt_point['Friction (€)']], y=[opt_point['Benefit (€)']], mode='markers', name='Optimal Setup (Peak NFI)', marker=dict(color='orange', size=14, symbol='star')))
fig.add_trace(go.Scatter(x=[curr_friction], y=[curr_benefit], mode='markers', name='Current Rules State', marker=dict(color='red', size=12, symbol='x')))

fig.update_layout(title='Efficient Frontier: Where we are vs. Where we could be', xaxis_title='Cost of Friction (Lost Margin + Churn LTV) €', yaxis_title='Fraud & Fees Prevented €', hovermode="x unified", legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95))
st.plotly_chart(fig, use_container_width=True)

# --- NEW: MONTHLY MARKET BENCHMARK TABLE ---
st.subheader("Market & Monthly Threshold Benchmarks")
benchmark_data = []
groups = df.groupby(['global_entity_id', 'Month'])

for (mkt, mth), group_df in groups:
    if len(group_df) < 50: continue 
    _, opt_row, act_thresh = calculate_frontier(group_df, margin_pct, cb_fee, ltv_cost)
    diff = opt_row['Threshold'] - act_thresh
    status = "Target Met"
    if diff > 5: status = "Too Stiff (Over-blocking)"
    elif diff < -5: status = "Too Loose (Under-blocking)"
        
    benchmark_data.append({
        'Market': mkt, 'Month': mth, 'Optimal Threshold': round(opt_row['Threshold'], 1),
        'Equivalent Actual Threshold': round(act_thresh, 1), 'Difference': round(diff, 1), 'Status': status
    })

if benchmark_data:
    df_bench = pd.DataFrame(benchmark_data).sort_values(by=['Market', 'Month'])
    def color_status(val): return f"color: {'red' if 'Over' in val else 'orange' if 'Under' in val else 'green'}"
    st.dataframe(df_bench.style.map(color_status, subset=['Status']), use_container_width=True, hide_index=True)

# --- RULE LEVEL ATTRIBUTION TABLE (WITH ACCURACY METRICS) ---
st.subheader("Granular Rule Performance & Accuracy")
rule_stats = []
active_rules = df[df['rule_name'] != 'None']['rule_name'].unique()

for rule in active_rules:
    r_df = df[df['rule_name'] == rule]
    if r_df.empty: continue
    
    tp_df = r_df[(r_df['is_actual_chargeback'] == 1) & (r_df['order_final_status'] == 'CANCELLED')]
    benefit = tp_df['amount_eur'].sum() + (len(tp_df) * cb_fee)
    
    fp_df = r_df[(r_df['is_actual_chargeback'] == 0) & (r_df['order_final_status'] == 'CANCELLED')]
    friction = (fp_df['amount_eur'].sum() * margin_pct) + (len(fp_df[fp_df['payment_switch'] == 0]) * ltv_cost)
    
    # Calculate accuracy metrics
    rule_capture = (tp_df['amount_eur'].sum() / actual_fraud_gmv) * 100 if actual_fraud_gmv > 0 else 0
    rule_precision = (len(tp_df) / (len(tp_df) + len(fp_df))) * 100 if (len(tp_df) + len(fp_df)) > 0 else 0
    friction_ratio = f"1 : {(len(fp_df)/len(tp_df)):.1f}" if len(tp_df) > 0 else "∞ (All FPs)"
    
    rule_stats.append({
        'Rule Name': rule,
        'Action': r_df['rule_action'].iloc[0],
        'Capture Rate (%)': rule_capture,
        'Precision (%)': rule_precision,
        'Friction Ratio (FP:TP)': friction_ratio,
        'TPs (Caught Fraud)': len(tp_df),
        'FPs (Lost Users)': len(fp_df),
        'Net Impact (€)': benefit - friction
    })

if rule_stats:
    df_rules = pd.DataFrame(rule_stats).sort_values('Net Impact (€)', ascending=False)
    st.dataframe(df_rules.style.format({
        'Net Impact (€)': '€{:,.0f}',
        'Capture Rate (%)': '{:.2f}%',
        'Precision (%)': '{:.1f}%'
    }), use_container_width=True)
