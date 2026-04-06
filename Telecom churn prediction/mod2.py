import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# CONFIG & STYLE
# ------------------------------
st.set_page_config(page_title="Telecom Bill Analysis Dashboard", layout="wide")

st.markdown("""
<style>
    .analysis-card {
        border: 2px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }
    .analysis-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        border-color: #3b82f6;
    }
    .model-header {
        font-weight: 700;
        font-size: 1.2em;
        color: #1e293b;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .badge-anomaly {background: linear-gradient(45deg, #f59e0b, #d97706); color:white; padding:4px 12px; border-radius:20px; font-size:0.75em; font-weight:600;}
    .badge-matching {background: linear-gradient(45deg, #10b981, #059669); color:white; padding:4px 12px; border-radius:20px; font-size:0.75em; font-weight:600;}
    .badge-fraud    {background: linear-gradient(45deg, #ef4444, #dc2626); color:white; padding:4px 12px; border-radius:20px; font-size:0.75em; font-weight:600;}
    .badge-valued   {background: linear-gradient(45deg, #6366f1, #4f46e5); color:white; padding:4px 12px; border-radius:20px; font-size:0.75em; font-weight:600;}
    .score-highlight {font-size:2em; font-weight:800; color:#059669; margin:8px 0;}
    .risk-high {color:#ef4444;}
    .risk-medium {color:#f59e0b;}
    .risk-low {color:#10b981;}
    .filter-section {background: linear-gradient(135deg, #f8fafc, #e2e8f0); border-radius:12px; padding:16px; margin:8px 0;}
    .profile-box {background: linear-gradient(135deg, #f0f9ff, #e0f2fe); border:2px solid #0ea5e9; border-radius:16px; padding:20px; box-shadow:0 4px 12px rgba(14,116,144,0.1); margin:16px 0;}
    .bulk-stats {background: linear-gradient(135deg, #030e06, rgb(38, 39, 48)); border:2px solid #22c55e; border-radius:12px; padding:16px; margin-top:16px;}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# API CONFIG
# ------------------------------
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-3.5-turbo"

# ------------------------------
# DATA LOADING & CLEANING
# ------------------------------
uploaded_file = st.file_uploader("Upload your bill CSV file", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_data():
        df = pd.read_csv(uploaded_file)

        # Rename to canonical schema
        df = df.rename(columns={
            'month': 'invoice_month',
            'invoice_total_GBP': 'bill_amount',
            'payment_received_GBP': 'payment_amount',
            'data_used_gb': 'data_usage_gb'
        })

        # Derive dispute flag
        df['dispute_flag'] = df['dispute_description'].notna() | df['dispute_type'].notna()

        # Type handling
        numeric_cols = ["bill_amount", "payment_amount", "data_usage_gb",
                        "base_plan_charge_GBP", "overage_charges_expected_GBP",
                        "overage_charges_billed_GBP", "tax_GBP", "reconciliation_match_score"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df['dispute_flag'] = df['dispute_flag'].astype(bool)
        df['invoice_month'] = pd.to_datetime(df['invoice_month'], errors='coerce')
        df = df.sort_values(['customer_id', 'invoice_month'])

        return df

    df_bills = load_data()

    # Compute stats for categorisation and dashboard
    total_customers = len(df_bills['customer_id'].unique())

    anom_cust = set(df_bills[df_bills.get('billing_anomaly_flag', False) == True]['customer_id']) if 'billing_anomaly_flag' in df_bills.columns else set()
    num_anom_cust = len(anom_cust)
    anom_list = list(anom_cust)

    mis_bills = df_bills[df_bills.get('reconciliation_match_score', 1) < 0.8][['customer_id', 'invoice_month', 'reconciliation_match_score']] if 'reconciliation_match_score' in df_bills.columns else pd.DataFrame(columns=['customer_id', 'invoice_month', 'reconciliation_match_score'])
    num_mis_bills = len(mis_bills)
    mis_cust = set(mis_bills['customer_id'])

    disp_bills = df_bills[df_bills['dispute_flag'] == True][['customer_id', 'invoice_month', 'dispute_type']]
    num_disp_bills = len(disp_bills)
    disp_cust = set(disp_bills['customer_id'])

    bad = anom_cust | mis_cust | disp_cust
    valued_cust = [c for c in df_bills['customer_id'].unique() if c not in bad]
    num_valued_cust = len(valued_cust)

    # Additional aggregations for charts
    monthly_bills = df_bills.groupby(df_bills['invoice_month'].dt.to_period('M')).agg({
        'bill_amount': 'sum',
        'customer_id': 'nunique',
        'dispute_flag': 'sum'
    }).reset_index()
    monthly_bills['invoice_month'] = monthly_bills['invoice_month'].dt.to_timestamp()

    bill_dist = pd.cut(df_bills['bill_amount'], bins=[0, 50, 100, 200, 500, 1000, float('inf')], 
                       labels=['£0-50', '£50-100', '£100-200', '£200-500', '£500-1000', '£1000+']).value_counts().reset_index()
    bill_dist.columns = ['Bill Range', 'Count']

    # Dispute vs Non-Dispute over time
    monthly_dispute_status = df_bills.groupby([df_bills['invoice_month'].dt.to_period('M'), 'dispute_flag']).size().unstack(fill_value=0).reset_index()
    monthly_dispute_status.columns = ['Invoice Month', 'No Dispute', 'Dispute']
    monthly_dispute_status['Invoice Month'] = monthly_dispute_status['Invoice Month'].dt.to_timestamp()
    if 'No Dispute' not in monthly_dispute_status.columns:
        monthly_dispute_status['No Dispute'] = 0
    if 'Dispute' not in monthly_dispute_status.columns:
        monthly_dispute_status['Dispute'] = 0

    # Customer distribution by status
    status_counts = {
        'Anomalous': num_anom_cust,
        'Mismatched': len(mis_cust),
        'Disputed': len(disp_cust),
        'Valued': num_valued_cust
    }
    status_df = pd.DataFrame(list(status_counts.items()), columns=['Status', 'Count'])

else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# ------------------------------
# LLM ANALYSIS
# ------------------------------
def analyze_bill_with_llm(record):
    prompt_template = """You are a telecom billing expert. Analyze this bill record:

{record_json}

Provide three model outputs (keep reasons <100 chars):

1. Anomaly Detection – score (0-1), is_anomalous (bool), reason
2. Matching Model – match_score (0-1), discrepancy_amount, reason
3. Fraud Dispute Classifier – fraud_probability (0-1), classification ('fraud'|'legitimate'|'no_dispute'), reason

Return **only** valid JSON:
{{
  "anomaly_detection": {{"score":0.0,"is_anomalous":false,"reason":""}},
  "matching_model": {{"match_score":1.0,"discrepancy_amount":0.0,"reason":""}},
  "fraud_dispute_classifier": {{"fraud_probability":0.0,"classification":"no_dispute","reason":""}}
}}"""

    record_json = json.dumps(record, indent=2, default=str)
    prompt = prompt_template.format(record_json=record_json)

    try:
        resp = requests.post(
            OPENROUTER_API_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.2, "max_tokens": 800},
            timeout=30
        )
        resp.raise_for_status()
        txt = resp.json()["choices"][0]["message"]["content"].strip()
        if txt.startswith("```json"): txt = txt[7:-3].strip()
        return json.loads(txt)
    except Exception as e:
        return {"error": str(e)}

# ------------------------------
# HELPERS
# ------------------------------
def get_customer_summary(cust_id):
    sub = df_bills[df_bills['customer_id'] == cust_id]
    return {
        'avg_bill': sub['bill_amount'].mean(),
        'total_bills': len(sub),
        'dispute_rate': sub['dispute_flag'].mean(),
        'avg_usage': sub['data_usage_gb'].mean()
    }

def render_card(title, result, badge_class):
    if "error" in result:
        return f'<div class="analysis-card"><p style="color:red;">Error: {result["error"]}</p></div>'

    key_map = {"Anomaly Detection Model": "anomaly_detection",
               "Matching Model": "matching_model",
               "Fraud Dispute Classifier": "fraud_dispute_classifier",
               "Valued Customer": "valued_customer"}

    model_res = result.get(key_map[title], {})
    score = model_res.get('score',
              model_res.get('match_score',
              model_res.get('fraud_probability', 0)))
    risk = 'risk-high' if score > 0.7 else ('risk-medium' if score > 0.3 else 'risk-low')
    reason = model_res.get('reason', 'N/A')
    cls = model_res.get('classification', '')

    return f"""
    <div class="analysis-card">
        <div class="model-header"><span class="{badge_class}">{title}</span></div>
        <div class="score-highlight {risk}">{score:.2f}</div>
        {f'<div style="font-weight:600;color:#ef4444;">{cls}</div>' if cls else ''}
        <p><strong>Reason:</strong> {reason}</p>
    </div>
    """

def format_column_names(df):
    formatted_cols = {col: ' '.join(word.capitalize() for word in col.split('_')) for col in df.columns}
    return df.rename(columns=formatted_cols)

# ------------------------------
# MAIN UI
# ------------------------------
st.title("Telecom Bill Analysis Dashboard")
st.markdown("**AI-powered anomaly detection, bill-matching, fraud classification & valued-customer recognition**")

# ---- Data Preview (no stats) ----
with st.expander("Dataset Preview", expanded=True):
    preview_df = df_bills.sample(n=min(10, len(df_bills))).copy()
    preview_df = format_column_names(preview_df)
    st.dataframe(
        preview_df.style.format({
            'Bill Amount': '£{:.2f}',
            'Data Usage Gb': '{:.1f}',
            'Payment Amount': '£{:.2f}'
        }).background_gradient(cmap='viridis', subset=['Bill Amount']),
        height=400, use_container_width=True
    )

# ---- Dataset Statistics ----
st.subheader("Dataset Statistics")
stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
with stat_col1:
    st.metric("Total Customers", total_customers)
with stat_col2:
    st.metric("Anomalous Customers", num_anom_cust)
with stat_col3:
    st.metric("Mismatched Bills", num_mis_bills)
with stat_col4:
    st.metric("Disputed Bills", num_disp_bills)
with stat_col5:
    st.metric("Valued Customers", num_valued_cust)

# ---- Visual Charts ----
st.subheader("Data Insights & Visualizations")

# Row 1: Pie + Bar
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Customer Status Distribution**")
    fig_pie = px.pie(status_df, values='Count', names='Status', 
                     color_discrete_sequence=px.colors.sequential.Plasma,
                     hole=0.4)
    fig_pie.update_layout(showlegend=True, height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

with chart_col2:
    st.markdown("**Bill Amount Distribution**")
    fig_bar = px.bar(bill_dist, x='Bill Range', y='Count', 
                     color='Bill Range', text='Count',
                     color_discrete_sequence=px.colors.sequential.Viridis)
    fig_bar.update_layout(showlegend=False, height=400, xaxis_title="Bill Range", yaxis_title="Number of Bills")
    st.plotly_chart(fig_bar, use_container_width=True)

# Row 2: Line (Dispute vs Non-Dispute) + Bar (Customer Distribution)
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.markdown("**Dispute vs Non-Dispute Trend Over Time**")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=monthly_dispute_status['Invoice Month'],
        y=monthly_dispute_status['No Dispute'],
        mode='lines+markers',
        name='No Dispute',
        line=dict(color='#10b981', width=3),
        fill='tozeroy'
    ))
    fig_line.add_trace(go.Scatter(
        x=monthly_dispute_status['Invoice Month'],
        y=monthly_dispute_status['Dispute'],
        mode='lines+markers',
        name='Dispute',
        line=dict(color='#ef4444', width=3),
        fill='tonexty'
    ))
    fig_line.update_layout(
        height=400,
        title_text="Dispute vs Non-Dispute Bills",
        xaxis_title="Month",
        yaxis_title="Number of Bills",
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )
    st.plotly_chart(fig_line, use_container_width=True)

with chart_col4:
    st.markdown("**Customer Distribution by Status**")
    fig_status_bar = px.bar(
        status_df, x='Status', y='Count',
        color='Status', text='Count',
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    fig_status_bar.update_layout(
        showlegend=False, height=400,
        xaxis_title="Customer Status",
        yaxis_title="Number of Customers"
    )
    st.plotly_chart(fig_status_bar, use_container_width=True)

# ------------------------------
# CUSTOMER CATEGORISATION WITH CONTACT INFO
# ------------------------------
st.subheader("Customer Categorisation")

# Helper: Add contact info to customer lists
def enrich_with_contact(df, cust_col='customer_id'):
    contact_cols = ['customer_name', 'uk_contact_number', 'email']
    base = df[[cust_col]].drop_duplicates()
    contact_info = df_bills[df_bills[cust_col].isin(base[cust_col])][[cust_col] + contact_cols].drop_duplicates(cust_col)
    return base.merge(contact_info, on=cust_col, how='left')

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# 1. Anomaly
with row1_col1:
    st.markdown("**Anomaly Detected**")
    if 'billing_anomaly_flag' in df_bills.columns and num_anom_cust > 0:
        anom_df = enrich_with_contact(pd.DataFrame({'customer_id': anom_list}))
        sel_anom = st.selectbox("Select for reason", anom_df['customer_id'], key="anom_sel")
        if sel_anom:
            rec = df_bills[(df_bills['customer_id'] == sel_anom) &
                          (df_bills['billing_anomaly_flag'] == True)].iloc[0].to_dict()
            res = analyze_bill_with_llm(rec)
            st.write("**Reason:**", res['anomaly_detection'].get('reason','N/A'))
        anom_display = format_column_names(anom_df)
        st.dataframe(anom_display, use_container_width=True)
    else:
        st.write("—")

# 2. Mismatched
with row1_col2:
    st.markdown("**Mismatched Bills**")
    if 'reconciliation_match_score' in df_bills.columns and num_mis_bills > 0:
        mis_with_contact = mis_bills.merge(
            df_bills[['customer_id', 'customer_name', 'uk_contact_number', 'email']].drop_duplicates(),
            on='customer_id', how='left'
        )
        mis_with_contact['Id'] = mis_with_contact['customer_id'] + ' - ' + mis_with_contact['invoice_month'].dt.strftime('%Y-%m')
        sel_mis = st.selectbox("Select for reason", mis_with_contact['Id'], key="mis_sel")
        if sel_mis:
            cust, mon = sel_mis.split(' - ')
            rec = df_bills[(df_bills['customer_id'] == cust) &
                          (df_bills['invoice_month'].dt.strftime('%Y-%m') == mon)].iloc[0].to_dict()
            res = analyze_bill_with_llm(rec)
            st.write("**Reason:**", res['matching_model'].get('reason','N/A'))
        mis_display = mis_with_contact[['customer_id', 'invoice_month', 'reconciliation_match_score', 'customer_name', 'uk_contact_number', 'email']]
        mis_display = format_column_names(mis_display)
        st.dataframe(mis_display.drop('Id', axis=1, errors='ignore'), use_container_width=True)
    else:
        st.write("—")

# 3. Disputed
with row2_col1:
    st.markdown("**Disputed Bills**")
    if num_disp_bills > 0:
        disp_with_contact = disp_bills.merge(
            df_bills[['customer_id', 'customer_name', 'uk_contact_number', 'email']].drop_duplicates(),
            on='customer_id', how='left'
        )
        disp_with_contact['Id'] = disp_with_contact['customer_id'] + ' - ' + disp_with_contact['invoice_month'].dt.strftime('%Y-%m')
        sel_disp = st.selectbox("Select for reason", disp_with_contact['Id'], key="disp_sel")
        if sel_disp:
            cust, mon = sel_disp.split(' - ')
            rec = df_bills[(df_bills['customer_id'] == cust) &
                      (df_bills['invoice_month'].dt.strftime('%Y-%m') == mon)].iloc[0].to_dict()
            res = analyze_bill_with_llm(rec)
            st.write("**Reason:**", res['fraud_dispute_classifier'].get('reason','N/A'))
        disp_display = disp_with_contact[['customer_id', 'invoice_month', 'dispute_type', 'customer_name', 'uk_contact_number', 'email']]
        disp_display = format_column_names(disp_display)
        st.dataframe(disp_display.drop('Id', axis=1, errors='ignore'), use_container_width=True)
    else:
        st.write("—")

# 4. Valued Customers
with row2_col2:
    st.markdown("**Valued Customers**")
    if num_valued_cust > 0:
        valued_df = enrich_with_contact(pd.DataFrame({'customer_id': valued_cust}))
        sel_val = st.selectbox("Select for reason", valued_df['customer_id'], key="val_sel")
        if sel_val:
            rec = df_bills[(df_bills['customer_id'] == sel_val) &
                          (~df_bills['customer_id'].isin(bad))].iloc[0].to_dict()
            prompt = f"""Customer {sel_val} has no anomaly, mismatch or dispute flags.
Explain in <80 chars why they are a **valued customer**."""
            try:
                resp = requests.post(
                    OPENROUTER_API_URL,
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type":"application/json"},
                    json={"model":OPENROUTER_MODEL,"messages":[{"role":"user","content":prompt}],
                          "temperature":0.1,"max_tokens":80},
                    timeout=20
                )
                reason = resp.json()["choices"][0]["message"]["content"].strip()
            except:
                reason = "Consistent billing, no issues."
            st.write("**Reason:**", reason)
        valued_display = format_column_names(valued_df)
        st.dataframe(valued_display, use_container_width=True)
    else:
        st.write("—")

# ---- Detailed Analysis (filters) - NO TABS ----
st.subheader("Detailed Bill Analysis")
fcol1, fcol2 = st.columns(2)
with fcol1:
    sel_cust = st.selectbox("Customer ID", df_bills['customer_id'].unique(), key="det_cust")
with fcol2:
    months = sorted(df_bills['invoice_month'].dt.to_period('M').unique())
    sel_mon = st.selectbox("Month", months, format_func=lambda x: x.strftime('%Y-%m'), key="det_mon")

rec_df = df_bills[(df_bills['customer_id'] == sel_cust) &
                 (df_bills['invoice_month'].dt.to_period('M') == sel_mon)]
selected_rec = rec_df.iloc[0].to_dict() if not rec_df.empty else {}

if selected_rec:
    # Customer profile
    summ = get_customer_summary(sel_cust)
    cust_info = df_bills[df_bills['customer_id'] == sel_cust][['customer_name', 'uk_contact_number', 'email']].iloc[0]
    st.markdown(f"""
    <div class="profile-box">
        <div style="font-size:1.3em;font-weight:700;color:#0c4a6b;margin-bottom:16px;">Customer {sel_cust}</div>
        <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px dashed #bae6fd;color:#0369a1;">
            <span style="font-weight:600;">Name</span><span>{cust_info['customer_name']}</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px dashed #bae6fd;color:#0369a1;">
            <span style="font-weight:600;">UK Contact</span><span>{cust_info['uk_contact_number']}</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px dashed #bae6fd;color:#0369a1;">
            <span style="font-weight:600;">Email</span><span>{cust_info['email']}</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:8px 0;color:#0369a1;">
            <span style="font-weight:600;">Avg Bill</span><span>£{summ['avg_bill']:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Recent 5 bills"):
        recent = df_bills[df_bills['customer_id']==sel_cust][['invoice_month','bill_amount','data_usage_gb','dispute_flag']].tail(5)
        recent = format_column_names(recent)
        st.dataframe(recent.style.format({'Bill Amount':'£{:.2f}','Data Usage Gb':'{:.1f}'}))

    # REMOVED: Tabs for Anomaly, Matching, Fraud

# ---- Bulk analysis for selected customer (KEEP THIS) ----
if st.button("Analyze **all** bills of selected customer", type="secondary", use_container_width=True):
    sub = df_bills[df_bills['customer_id'] == sel_cust]
    out = []
    for _, r in sub.iterrows():
        res = analyze_bill_with_llm(r.to_dict())
        flat = {
            'anomaly_score': res.get('anomaly_detection',{}).get('score',0),
            'match_score'  : res.get('matching_model',{}).get('match_score',0),
            'fraud_prob'   : res.get('fraud_dispute_classifier',{}).get('fraud_probability',0)
        }
        out.append({**r.to_dict(), **flat})
    st.session_state.bulk = pd.DataFrame(out)
    st.success(f"Processed {len(out)} bills")

# ---- Bulk Results + Stats ----
if "bulk" in st.session_state:
    st.subheader("Bulk Results")
    bulk_df = st.session_state.bulk[['invoice_month','bill_amount','anomaly_score','match_score','fraud_prob',
                                    'customer_name','uk_contact_number','email']]
    bulk_df = format_column_names(bulk_df)
    st.dataframe(bulk_df, use_container_width=True)

    # --- Bulk Stats ---
    total_bills = len(bulk_df)
    total_bill_amount = st.session_state.bulk['bill_amount'].sum()
    avg_anomaly = st.session_state.bulk['anomaly_score'].mean()
    avg_match = st.session_state.bulk['match_score'].mean()
    avg_fraud = st.session_state.bulk['fraud_prob'].mean()

    st.markdown(f"""
    <div class="bulk-stats">
        <div style="font-weight:700; font-size:1.2em; margin-bottom:12px; color:#166534;">
            Bulk Analysis Summary
        </div>
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; font-size:0.95em;">
            <div><strong>Total Bill Amount:</strong> <span style="color:#15803d;">£{total_bill_amount:,.2f}</span></div>
            <div><strong>Anomaly Score / {total_bills}:</strong> <span style="color:#ca8a04;">{avg_anomaly:.3f}</span></div>
            <div><strong>Match Score / {total_bills}:</strong> <span style="color:#16a34a;">{avg_match:.3f}</span></div>
            <div><strong>Fraud Prob / {total_bills}:</strong> <span style="color:#dc2626;">{avg_fraud:.3f}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("**Data:** Uploaded CSV | **AI:** OpenRouter GPT-3.5 | **Charts:** Plotly")