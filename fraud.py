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

# --- DYNAMIC KPIs ---
col1, col2, col3, col4 = st.columns(4)
total_gmv = df['amount_eur'].sum()
actual_fraud_gmv = df[df['is_actual_chargeback'] == 1]['amount_eur'].sum()
col1.metric("Total Processed GMV", f"€{total_gmv:,.0f}")
col2.metric("Underlying Fraud Exposure", f"€{actual_fraud_gmv:,.0f}")

# --- HELPER FUNCTION: CALCULATE FRONTIER & ACTUALS ---
def calculate_frontier(df_subset, margin, fee, ltv):
    thresholds = np.linspace(5, 95, 40)
    curve = []
    
    # 1. Current State (Hardcoded Rules)
    curr_tp = df_subset[(df_subset['rule_name'] != 'None') & (df_subset['order_final_status'] == 'CANCELLED') & (df_subset['is_actual_chargeback'] == 1)]
    curr_benefit = curr_tp['amount_eur'].sum() + (len(curr_tp) * fee)
    
    curr_fp = df_subset[(df_subset['rule_name'] != 'None') & (df_subset['order_final_status'] == 'CANCELLED') & (df_subset['is_actual_chargeback'] == 0)]
    curr_margin_loss = curr_fp['amount_eur'].sum() * margin
    curr_friction = curr_margin_loss + (len(curr_fp[curr_fp['payment_switch'] == 0]) * ltv)
    
    # 2. Simulated ML Frontier
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
    
    # Calculate "Equivalent Actual Threshold" based on closest Friction match
    df_c['Friction_Diff'] = abs(df_c['Friction (€)'] - curr_friction)
    actual_eq_row = df_c.loc[df_c['Friction_Diff'].idxmin()]
    
    return df_c, opt_row, curr_benefit, curr_friction, actual_eq_row['Threshold']

# --- MAIN CALCULATIONS ---
df_curve, opt_point, curr_benefit, curr_friction, actual_eq_thresh = calculate_frontier(df, margin_pct, cb_fee, ltv_cost)

col3.metric("Max Net Financial Impact", f"€{opt_point['NFI']:,.0f}")
col4.metric("Optimal Risk Threshold", f"{opt_point['Threshold']:.1f}")

# --- PLOT EFFICIENT FRONTIER ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_curve['Friction (€)'], y=df_curve['Benefit (€)'], mode='lines', name='Risk Score Frontier', line=dict(color='royalblue', width=3)))
fig.add_trace(go.Scatter(x=[opt_point['Friction (€)']], y=[opt_point['Benefit (€)']], mode='markers', name='Optimal Setup (Peak NFI)', marker=dict(color='orange', size=14, symbol='star')))
fig.add_trace(go.Scatter(x=[curr_friction], y=[curr_benefit], mode='markers', name='Current Rules State', marker=dict(color='red', size=12, symbol='x')))

fig.update_layout(title='Efficient Frontier: Where we are vs. Where we could be', xaxis_title='Cost of Friction (Lost Margin + Churn LTV) €', yaxis_title='Fraud & Fees Prevented €', hovermode="x unified", legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95))
st.plotly_chart(fig, use_container_width=True)

# --- NEW: MONTHLY MARKET BENCHMARK TABLE ---
st.subheader("Market & Monthly Threshold Benchmarks")
st.markdown("Compares the mathematical Optimal Threshold against the Equivalent Actual Threshold (based on the friction your hardcoded rules are currently causing).")

benchmark_data = []
groups = df.groupby(['global_entity_id', 'Month'])

for (mkt, mth), group_df in groups:
    # Skip groups that are too small to build a meaningful curve
    if len(group_df) < 50: continue 
    
    _, opt_row, _, _, act_thresh = calculate_frontier(group_df, margin_pct, cb_fee, ltv_cost)
    
    # If the Actual Equivalent is strictly lower than optimal, we are "Over-blocking" (too strict)
    diff = opt_row['Threshold'] - act_thresh
    status = "Target Met"
    if diff > 5: status = "Too Stiff (Over-blocking)"
    elif diff < -5: status = "Too Loose (Under-blocking)"
        
    benchmark_data.append({
        'Market': mkt,
        'Month': mth,
        'Optimal Threshold': round(opt_row['Threshold'], 1),
        'Equivalent Actual Threshold': round(act_thresh, 1),
        'Difference': round(diff, 1),
        'Status': status
    })

if benchmark_data:
    df_bench = pd.DataFrame(benchmark_data).sort_values(by=['Market', 'Month'])
    
    # Formatting for Streamlit display
    def color_status(val):
        color = 'red' if 'Over' in val else 'orange' if 'Under' in val else 'green'
        return f'color: {color}'
        
    st.dataframe(df_bench.style.map(color_status, subset=['Status']), use_container_width=True, hide_index=True)
else:
    st.info("Not enough data to generate monthly benchmarks with the current filters.")

# --- RULE LEVEL ATTRIBUTION TABLE (Existing) ---
st.subheader("Granular Rule Performance")
rule_stats = []
active_rules = df[df['rule_name'] != 'None']['rule_name'].unique()

for rule in active_rules:
    r_df = df[df['rule_name'] == rule]
    if r_df.empty: continue
    tp_df = r_df[(r_df['is_actual_chargeback'] == 1) & (r_df['order_final_status'] == 'CANCELLED')]
    benefit = tp_df['amount_eur'].sum() + (len(tp_df) * cb_fee)
    fp_df = r_df[(r_df['is_actual_chargeback'] == 0) & (r_df['order_final_status'] == 'CANCELLED')]
    friction = (fp_df['amount_eur'].sum() * margin_pct) + (len(fp_df[fp_df['payment_switch'] == 0]) * ltv_cost)
    
    rule_stats.append({
        'Rule Name': rule,
        'Action': r_df['rule_action'].iloc[0],
        'Triggers': len(r_df),
        'TPs (Caught Fraud)': len(tp_df),
        'FPs (Lost Users)': len(fp_df),
        'Net Impact (€)': benefit - friction
    })

if rule_stats:
    df_rules = pd.DataFrame(rule_stats).sort_values('Net Impact (€)', ascending=False)
    st.dataframe(df_rules.style.format({'Net Impact (€)': '€{:,.0f}'}), use_container_width=True)
