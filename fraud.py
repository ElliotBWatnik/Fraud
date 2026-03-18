import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Optimization Engine", layout="wide")

# --- DATA LOADING & PREP ---
@st.cache_data
def load_data():
    df = pd.read_csv('fraud_data.csv')
    # Ensure dates are parsed and extract the Month for filtering
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['Month'] = df['order_date'].dt.strftime('%Y-%m')
    return df

df_raw = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("1. Global Filters")

# Single select boxes with "All" as the default
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

if market != 'All Markets':
    df = df[df['global_entity_id'] == market]
if month != 'All Months':
    df = df[df['Month'] == month]
if vertical != 'All Verticals':
    df = df[df['vendor_vertical'] == vertical]
    
if is_saved == 'Saved Card Only':
    df = df[df['is_saved_card'] == True]
elif is_saved == 'New Card Only':
    df = df[df['is_saved_card'] == False]

st.title("🛡️ Fraud Prevention Efficient Frontier")

# Stop execution if filters result in empty dataframe
if df.empty:
    st.warning("No data available for this combination of filters. Please adjust your selections.")
    st.stop()

# --- DYNAMIC KPIs ---
col1, col2, col3, col4 = st.columns(4)
total_gmv = df['amount_eur'].sum()
actual_fraud_gmv = df[df['is_actual_chargeback'] == 1]['amount_eur'].sum()
col1.metric("Total Processed GMV", f"€{total_gmv:,.0f}")
col2.metric("Underlying Fraud Exposure", f"€{actual_fraud_gmv:,.0f}", f"{(actual_fraud_gmv/total_gmv)*100:.2f}% of GMV", delta_color="inverse")

# --- CALCULATE THE FRONTIER (SIMULATED ML SCORE) ---
thresholds = np.linspace(5, 95, 40)
curve_data = []

for t in thresholds:
    df_sim = df.copy()
    df_sim['sim_blocked'] = df_sim['risk_score'] >= t
    
    # True Positives
    tp = df_sim[(df_sim['sim_blocked']) & (df_sim['is_actual_chargeback'] == 1)]
    benefit = tp['amount_eur'].sum() + (len(tp) * cb_fee)
    
    # False Positives
    fp = df_sim[(df_sim['sim_blocked']) & (df_sim['is_actual_chargeback'] == 0)]
    margin_loss = fp['amount_eur'].sum() * margin_pct
    churned_fp = fp[fp['payment_switch'] == 0] 
    ltv_loss = len(churned_fp) * ltv_cost
    friction = margin_loss + ltv_loss
    
    curve_data.append({'Threshold': t, 'Benefit (€)': benefit, 'Friction (€)': friction, 'NFI': benefit - friction})

df_curve = pd.DataFrame(curve_data)
optimal_idx = df_curve['NFI'].idxmax()
opt_point = df_curve.iloc[optimal_idx]

col3.metric("Max Net Financial Impact", f"€{opt_point['NFI']:,.0f}")
col4.metric("Optimal Risk Threshold", f"{opt_point['Threshold']:.1f}")

# --- CALCULATE CURRENT STATE (HARDCODED RULES) ---
# How are the actual rules performing today?
curr_tp = df[(df['rule_name'] != 'None') & (df['order_final_status'] == 'CANCELLED') & (df['is_actual_chargeback'] == 1)]
curr_benefit = curr_tp['amount_eur'].sum() + (len(curr_tp) * cb_fee)

curr_fp = df[(df['rule_name'] != 'None') & (df['order_final_status'] == 'CANCELLED') & (df['is_actual_chargeback'] == 0)]
curr_margin_loss = curr_fp['amount_eur'].sum() * margin_pct
curr_churned_fp = curr_fp[curr_fp['payment_switch'] == 0]
curr_friction = curr_margin_loss + (len(curr_churned_fp) * ltv_cost)

# --- PLOT EFFICIENT FRONTIER ---
fig = go.Figure()

# 1. The Curve (What is mathematically possible)
fig.add_trace(go.Scatter(x=df_curve['Friction (€)'], y=df_curve['Benefit (€)'], 
                         mode='lines', name='Risk Score Frontier', line=dict(color='royalblue', width=3)))

# 2. The Optimal Point
fig.add_trace(go.Scatter(x=[opt_point['Friction (€)']], y=[opt_point['Benefit (€)']], 
                         mode='markers', name='Optimal Setup (Peak NFI)', 
                         marker=dict(color='orange', size=14, symbol='star')))

# 3. The Current State
fig.add_trace(go.Scatter(x=[curr_friction], y=[curr_benefit], 
                         mode='markers', name='Current Rules State', 
                         marker=dict(color='red', size=12, symbol='x')))

fig.update_layout(title='Efficient Frontier: Where we are vs. Where we could be',
                  xaxis_title='Cost of Friction (Lost Margin + Churn LTV) €',
                  yaxis_title='Fraud & Fees Prevented €',
                  hovermode="x unified",
                  legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95))

st.plotly_chart(fig, use_container_width=True)

# --- RULE LEVEL ATTRIBUTION TABLE ---
st.subheader(f"Granular Rule Performance")
st.markdown("Attribution of the 'Current Rules State' marker above.")

rule_stats = []
active_rules = df[df['rule_name'] != 'None']['rule_name'].unique()

for rule in active_rules:
    r_df = df[df['rule_name'] == rule]
    
    # Safe guard against empty slices to prevent IndexError
    if r_df.empty:
        continue
        
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
        'Benefit (€)': benefit,
        'Friction Cost (€)': friction,
        'Net Impact (€)': benefit - friction
    })

if rule_stats:
    df_rules = pd.DataFrame(rule_stats).sort_values('Net Impact (€)', ascending=False)
    st.dataframe(df_rules.style.format({
        'Benefit (€)': '€{:,.0f}', 
        'Friction Cost (€)': '€{:,.0f}', 
        'Net Impact (€)': '€{:,.0f}'
    }), use_container_width=True)
else:
    st.info("No active rules triggered for this filtered segment.")
