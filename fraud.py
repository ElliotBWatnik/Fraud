import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Optimization Engine", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    return pd.read_csv('fraud_data.csv')

df_raw = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("1. Global Filters")
market = st.sidebar.multiselect("Market Entity", options=df_raw['global_entity_id'].unique(), default=df_raw['global_entity_id'].unique())
vertical = st.sidebar.multiselect("Vendor Vertical", options=df_raw['vendor_vertical'].unique(), default=df_raw['vendor_vertical'].unique())
is_saved = st.sidebar.radio("Card Status", options=['All', 'Saved Card Only', 'New Card Only'])

st.sidebar.header("2. Economic Assumptions")
margin_pct = st.sidebar.slider("Net Profit Margin (%)", 1, 30, 10) / 100.0
cb_fee = st.sidebar.number_input("Chargeback Fee (€)", value=15.0)
ltv_cost = st.sidebar.number_input("Blended LTV Cost per Churn (€)", value=50.0)

# --- APPLY FILTERS ---
df = df_raw[df_raw['global_entity_id'].isin(market) & df_raw['vendor_vertical'].isin(vertical)]
if is_saved == 'Saved Card Only':
    df = df[df['is_saved_card'] == True]
elif is_saved == 'New Card Only':
    df = df[df['is_saved_card'] == False]

st.title("🛡️ Fraud Prevention Efficient Frontier")
st.markdown("Optimize the balance between blocking fraudsters and dropping legitimate customers.")

# --- DYNAMIC KPIs ---
col1, col2, col3, col4 = st.columns(4)
total_gmv = df['amount_eur'].sum()
actual_fraud_gmv = df[df['is_actual_chargeback'] == 1]['amount_eur'].sum()
col1.metric("Total Processed GMV", f"€{total_gmv:,.0f}")
col2.metric("Underlying Fraud Exposure", f"€{actual_fraud_gmv:,.0f}", f"{(actual_fraud_gmv/total_gmv)*100:.2f}% of GMV", delta_color="inverse")

# --- CALCULATE THE FRONTIER ---
# We simulate a sliding risk score threshold to draw the curve
thresholds = np.linspace(10, 95, 30)
curve_data = []

for t in thresholds:
    df_sim = df.copy()
    df_sim['sim_blocked'] = df_sim['risk_score'] >= t
    
    # True Positives
    tp = df_sim[(df_sim['sim_blocked']) & (df_sim['is_actual_chargeback'] == 1)]
    benefit = tp['amount_eur'].sum() + (len(tp) * cb_fee)
    
    # False Positives (Assuming drops if blocked, factoring in payment switch recovery)
    fp = df_sim[(df_sim['sim_blocked']) & (df_sim['is_actual_chargeback'] == 0)]
    margin_loss = fp['amount_eur'].sum() * margin_pct
    churned_fp = fp[fp['payment_switch'] == 0] # They didn't recover via another method
    ltv_loss = len(churned_fp) * ltv_cost
    friction = margin_loss + ltv_loss
    
    curve_data.append({'Threshold': t, 'Benefit (€)': benefit, 'Friction (€)': friction, 'NFI': benefit - friction})

df_curve = pd.DataFrame(curve_data)
optimal_idx = df_curve['NFI'].idxmax()
opt_point = df_curve.iloc[optimal_idx]

# Update KPIs with Optimal numbers
col3.metric("Max Net Financial Impact", f"€{opt_point['NFI']:,.0f}")
col4.metric("Optimal Risk Threshold", f"{opt_point['Threshold']:.1f}")

# --- PLOT EFFICIENT FRONTIER ---
fig = go.Figure()
# The Curve
fig.add_trace(go.Scatter(x=df_curve['Friction (€)'], y=df_curve['Benefit (€)'], 
                         mode='lines', name='Risk Score Frontier', line=dict(color='royalblue', width=3)))
# The Optimal Point
fig.add_trace(go.Scatter(x=[opt_point['Friction (€)']], y=[opt_point['Benefit (€)']], 
                         mode='markers', name='Optimal Strictness', 
                         marker=dict(color='red', size=12, symbol='star')))

fig.update_layout(title='Efficient Frontier (Fraud Prevented vs. Good User Cost)',
                  xaxis_title='Cost of Friction (Lost Margin + LTV) €',
                  yaxis_title='Fraud & Fees Prevented €',
                  hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- RULE LEVEL ATTRIBUTION TABLE ---
st.subheader("Granular Rule Performance (Current State)")
st.markdown("How your current hardcoded rules are performing based on the applied filters.")

rule_stats = []
active_rules = df[df['rule_name'] != 'None']['rule_name'].unique()

for rule in active_rules:
    r_df = df[df['rule_name'] == rule]
    
    # True Positives
    tp_df = r_df[(r_df['is_actual_chargeback'] == 1) & (r_df['order_final_status'] == 'CANCELLED')]
    benefit = tp_df['amount_eur'].sum() + (len(tp_df) * cb_fee)
    
    # False Positives
    fp_df = r_df[(r_df['is_actual_chargeback'] == 0) & (r_df['order_final_status'] == 'CANCELLED')]
    friction = (fp_df['amount_eur'].sum() * margin_pct) + (len(fp_df[fp_df['payment_switch'] == 0]) * ltv_cost)
    
    rule_stats.append({
        'Rule Name': rule,
        'Action': r_df['rule_action'].iloc[0],
        'Triggers': len(r_df),
        'FPs (Lost Users)': len(fp_df),
        'TPs (Caught Fraud)': len(tp_df),
        'Net Impact (€)': benefit - friction
    })

df_rules = pd.DataFrame(rule_stats).sort_values('Net Impact (€)', ascending=False)
st.dataframe(df_rules.style.format({'Net Impact (€)': '€{:,.0f}'}), use_container_width=True)
