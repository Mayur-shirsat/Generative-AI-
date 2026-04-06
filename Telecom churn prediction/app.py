import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# CONFIG & STYLE (MERGED FROM BOTH MODULES)
# ------------------------------
st.set_page_config(page_title="Smart Telecom Dashboard", layout="wide")

st.markdown("""
<style>
    .plan-card {
        border: 2px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
        position: relative;
        overflow: hidden;
    }
    .plan-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        border-color: #3b82f6;
    }
    .plan-header {
        font-weight: 700;
        font-size: 1.2em;
        color: #1e293b;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .badge-custom {
        background: linear-gradient(45deg, #ef4444, #dc2626);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75em;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(239,68,68,0.3);
    }
    .badge-standard {
        background: linear-gradient(45deg, #3b82f6, #2563eb);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75em;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(59,130,246,0.3);
    }
    .data-highlight {
        font-size: 1.4em;
        font-weight: 700;
        color: #059669;
        margin: 8px 0;
        background: linear-gradient(90deg, #059669, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .price-highlight {
        font-size: 1.6em;
        font-weight: 800;
        color: #16a34a;
        margin: 12px 0;
    }
    .features {
        flex-grow: 1;
        font-size: 0.9em;
        color: #64748b;
        line-height: 1.6;
        margin-bottom: auto;
    }
    .select-btn {
        background: linear-gradient(45deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95em;
        cursor: pointer;
        transition: all 0.2s;
        margin-top: 16px;
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
    }
    .select-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(59,130,246,0.4);
    }
    .recommend-box {
        background: linear-gradient(135deg, rgb(38, 39, 48), rgb(14, 17, 23));
        border-left: 5px solid #3b82f6;
        padding: 16px;
        border-radius: 0 12px 12px 0;
        margin: 16px 0;
    }
    .profile-box {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border: 2px solid #0ea5e9;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(14, 116, 144, 0.1);
        margin: 16px 0;
    }
    .profile-title {
        font-size: 1.3em;
        font-weight: 700;
        color: #0c4a6b;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .profile-metric {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px dashed #bae6fd;
        color: #0369a1;
        font-size: 0.95em;
    }
    .profile-metric:last-child {
        border-bottom: none;
    }
    .profile-label {
        font-weight: 600;
    }
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
    .bulk-stats {background: linear-gradient(135deg, #030e06, rgb(38, 39, 48)); border:2px solid #22c55e; border-radius:12px; padding:16px; margin-top:16px;}
    .main-title {
        font-size: 2.5em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 30px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# API CONFIG (SHARED)
# ------------------------------
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-3.5-turbo"

# ------------------------------
# STANDARD PLANS
# ------------------------------
standard_plans = {
    "Bronze 10GB": {"data": 10, "price": 12.00, "features": "• Unlimited minutes & texts\n• 4G/5G access\n• Basic perks & rewards"},
    "Silver 20GB": {"data": 20, "price": 18.99, "features": "• Unlimited minutes & texts\n• 4G/5G, 100Mbps max\n• Enhanced perks"},
    "Gold 50GB": {"data": 50, "price": 29.99, "features": "• Unlimited minutes & texts\n• Full 5G, unlimited speed\n• Premium perks"},
    "Platinum 100GB": {"data": 100, "price": 44.99, "features": "• Unlimited minutes & texts\n• 5G, VIP perks\n• Free add-ons"},
    "Diamond Unlimited": {"data": "Unlimited", "price": 59.99, "features": "• Unlimited everything\n• 5G+, priority support\n• Exclusive gifts"},
    "Family Bronze 10GB": {"data": 40, "price": 34.99, "features": "• Shared 40GB (up to 4 lines)\n• Unlimited mins/texts\n• Family basics"},
    "Family Silver 20GB": {"data": 80, "price": 52.99, "features": "• Shared 80GB (4 lines)\n• 100Mbps, enhanced family perks"},
    "Family Gold 50GB": {"data": 200, "price": 82.99, "features": "• Shared 200GB (4 lines)\n• Unlimited speed, premium family"},
    "Family Platinum 100GB": {"data": 400, "price": 119.99, "features": "• Shared 400GB (4 lines)\n• VIP family, free add-ons"},
    "Family Diamond Unlimited": {"data": "Unlimited", "price": 149.99, "features": "• Unlimited shared data\n• 5G+, priority family support"}
}

# ------------------------------
# MAIN TITLE
# ------------------------------
st.markdown('<h1 class="main-title">Telecom AI System</h1>', unsafe_allow_html=True)

# ------------------------------
# TABS
# ------------------------------
tab1, tab2 = st.tabs(["Plan Recommender", "Bill Analysis"])

# ==============================
# TAB 1: PLAN RECOMMENDER
# ==============================
with tab1:
    st.markdown("## Smart Telecom Plan Recommender")
    st.markdown("**AI analyzes your usage patterns and recommends perfect plans with data-backed justifications**")

    uploaded_file1 = st.file_uploader("Upload usage CSV file", type="csv", key="rec_upload")

    if uploaded_file1 is not None:
        @st.cache_data
        def load_and_clean_data(uploaded_file):
            df = pd.read_csv(uploaded_file)
            
            required_cols = ["customer_id", "month", "plan_name", "data_used_gb", "overage_charges_GBP"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                st.stop()
            
            df["month"] = pd.to_datetime(df["month"], errors='coerce')
            
            numeric_cols = ["data_used_gb", "overage_charges_GBP", "text_messages_sent", "call_minutes_used"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            def infer_limit(plan_name):
                for plan, info in standard_plans.items():
                    if plan in str(plan_name):
                        return info["data"] if info["data"] != "Unlimited" else np.inf
                return np.inf
            
            df["data_limit_gb"] = df["plan_name"].apply(infer_limit)
            df["tenure_years"] = pd.to_numeric(df.get("tenure_years", 1), errors='coerce').fillna(1).astype(int)
            df["satisfaction_score"] = pd.to_numeric(df.get("satisfaction_score", 8.0), errors='coerce').fillna(8.0)
            
            if "customer_name" in df.columns:
                df["customer_name"] = df["customer_name"].fillna("Unknown").astype(str)
            if "uk_contact_number" in df.columns:
                df["uk_contact_number"] = df["uk_contact_number"].fillna("N/A").astype(str)
            if "email" in df.columns:
                df["email"] = df["email"].fillna("N/A").astype(str)
            
            return df

        df_monthly = load_and_clean_data(uploaded_file1)

        with st.expander("Data Preview", expanded=True):
            display_df = df_monthly.head(15).copy()
            display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
            format_dict = {'Data Used Gb': '{:.1f}', 'Overage Charges Gbp': '£{:.2f}'}
            st.dataframe(
                display_df.style.format(format_dict).background_gradient(cmap='viridis', subset=['Data Used Gb']),
                height=400, use_container_width=True
            )
            
            st.subheader("Quick Stats Visualizations")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Average Data Used per Plan**")
                avg_data_per_plan = df_monthly.groupby('plan_name')['data_used_gb'].mean().sort_values()
                st.bar_chart(avg_data_per_plan)
            with col2:
                st.markdown("**Average Data Usage Over Time**")
                avg_data_over_time = df_monthly.groupby('month')['data_used_gb'].mean()
                st.line_chart(avg_data_over_time)
            with col3:
                st.markdown("**Plan Distribution**")
                plan_counts = df_monthly['plan_name'].value_counts().reset_index()
                plan_counts.columns = ['Plan Name', 'Count']
                fig = px.pie(plan_counts, values='Count', names='Plan Name', title='Plan Distribution')
                st.plotly_chart(fig, use_container_width=True)

        def compute_summary(df):
            summaries = []
            for cid, group in df.groupby("customer_id"):
                avg_data = group["data_used_gb"].mean()
                current_limit = group.iloc[-1]["data_limit_gb"]
                limit_val = np.inf if current_limit == np.inf else float(current_limit)
                months_over = (group["data_used_gb"] > 1.1 * limit_val).sum()
                avg_overage = group["overage_charges_GBP"].mean()
                avg_satisfaction = group["satisfaction_score"].mean()
                tenure = int(group.iloc[-1]["tenure_years"])
                current_plan = group.iloc[-1]["plan_name"]
                
                customer_name = group["customer_name"].iloc[0] if "customer_name" in group.columns else "Unknown"
                uk_contact_number = group["uk_contact_number"].iloc[0] if "uk_contact_number" in group.columns else "N/A"
                email = group["email"].iloc[0] if "email" in group.columns else "N/A"
                
                churn_risk = "Low"
                if avg_satisfaction < 6.0 or (months_over >= 6 and avg_overage > 2.5):
                    churn_risk = "High"
                elif avg_satisfaction < 7.0:
                    churn_risk = "Medium"
                
                summaries.append({
                    "customer_id": cid,
                    "customer_name": customer_name,
                    "uk_contact_number": uk_contact_number,
                    "email": email,
                    "current_plan": current_plan,
                    "avg_data_used_gb": round(avg_data, 2),
                    "data_limit_gb": "Unlimited" if limit_val == np.inf else round(limit_val, 1),
                    "months_over_110pct": int(months_over),
                    "avg_overage_GBP": round(avg_overage, 2),
                    "avg_satisfaction": round(avg_satisfaction, 1),
                    "tenure_years": tenure,
                    "churn_risk": churn_risk
                })
            return pd.DataFrame(summaries)

        if st.button("Analyze All Customers", type="primary", use_container_width=True):
            with st.spinner("Computing usage summaries..."):
                df_summary = compute_summary(df_monthly)
                st.session_state.df_summary = df_summary
                st.session_state.df_monthly = df_monthly
                st.success("Analysis complete!")
            
            st.dataframe(df_summary.head(10), use_container_width=True)
            st.download_button("Download Summaries", df_summary.to_csv(index=False).encode(), "customer_analysis.csv")

        if "df_summary" in st.session_state:
            df_summary = st.session_state.df_summary
            df_monthly = st.session_state.df_monthly
            
            st.subheader("Standard Plans")
            cols = st.columns(3)
            for idx, (name, info) in enumerate(standard_plans.items()):
                with cols[idx % 3]:
                    data_val = info["data"]
                    price_val = info["price"]
                    data_text = f"{data_val}GB Data" if data_val != "Unlimited" else "Unlimited Data"
                    price_text = f"£{price_val:.2f}<span style='font-size:0.7em;'>/month</span>"
                    badge_class = "badge-custom" if "Custom" in name else "badge-standard"
                    st.markdown(f"""
                    <div class="plan-card">
                        <div class="plan-header">
                            <span class="{badge_class}">{ 'CUSTOM' if 'Custom' in name else 'STANDARD'}</span>
                            {name}
                        </div>
                        <div class="data-highlight">{data_text}</div>
                        <div class="price-highlight" style="font-size:1.5em;">{price_text}</div>
                        <div class="features">{info['features'].replace('\\n', '<br>')}</div>
                        <button class="select-btn">Select This Plan</button>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.subheader("Your Personalized Recommendations")
            customer = st.selectbox("Select Customer", df_summary["customer_id"].tolist(), key="rec_cust")
            cust_summary = df_summary[df_summary["customer_id"] == customer].iloc[0].to_dict()
            
            contact_row = df_monthly[df_monthly["customer_id"] == customer].iloc[0]
            customer_name = contact_row.get("customer_name", "Unknown")
            uk_contact_number = contact_row.get("uk_contact_number", "N/A")
            email = contact_row.get("email", "N/A")
            
            st.markdown(f"""
            <div class="profile-box">
                <div class="profile-title">Customer Profile: {customer_name}</div>
                <div class="profile-metric">
                    <span class="profile-label">Customer ID</span>
                    <span>{cust_summary['customer_id']}</span>
                </div>
                <div class="profile-metric">
                    <span class="profile-label">Contact Number</span>
                    <span>{uk_contact_number}</span>
                </div>
                <div class="profile-metric">
                    <span class="profile-label">Email</span>
                    <span>{email}</span>
                </div>
                <div class="profile-metric">
                    <span class="profile-label">Current Plan</span>
                    <span>{cust_summary['current_plan']}</span>
                </div>
                <div class="profile-metric">
                    <span class="profile-label">Monthly Data Usage</span>
                    <span>{cust_summary['avg_data_used_gb']} GB</span>
                </div>
                <div class="profile-metric">
                    <span class="profile-label">Avg Overage Cost</span>
                    <span>£{cust_summary['avg_overage_GBP']}</span>
                </div>
                <div class="profile-metric">
                    <span class="profile-label">Churn Risk</span>
                    <span>{cust_summary['churn_risk']}</span>
                </div>
                <div class="profile-metric">
                    <span class="profile-label">Tenure</span>
                    <span>{cust_summary['tenure_years']} years</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Recent Usage"):
                recent = df_monthly[df_monthly["customer_id"] == customer][["month", "data_used_gb", "overage_charges_GBP"]].tail(6)
                recent.columns = ["Month", "Data Used (GB)", "Overage (£)"]
                st.dataframe(recent.style.format({'Data Used (GB)': '{:.1f}', 'Overage (£)': '£{:.2f}'}))

            def get_ai_recommendations(summary):
                discount_reasons = []
                discount = 0.0
                if summary['tenure_years'] >= 3:
                    discount += 0.10
                    discount_reasons.append("Loyalty (3+ years)")
                if summary['avg_satisfaction'] >= 8.0:
                    discount += 0.05
                    discount_reasons.append("High satisfaction")
                if summary['churn_risk'] == "High":
                    discount += 0.15
                    discount_reasons.append("High churn retention")
                elif summary['churn_risk'] == "Medium":
                    discount += 0.05
                    discount_reasons.append("Medium churn retention")
                if summary['months_over_110pct'] >= 3:
                    discount += 0.05
                    discount_reasons.append("Frequent overages")
                discount = min(discount, 0.30)
                
                discount_pct = int(discount * 100)
                disc_str = "; ".join(discount_reasons) or "Standard pricing"
                
                is_family = "Family" in summary['current_plan']
                avg_data = summary['avg_data_used_gb']
                step_size = 40 if is_family else 10
                family_base = 30 if is_family else 10
                
                custom_plans = []
                variants = [
                    ("Economy", 1.1, "4G/5G, Basic perks"),
                    ("Balanced", 1.2, "5G access, Standard perks"), 
                    ("Premium", 1.5, "5G+, Priority support, Premium perks")
                ]
                
                for name, buffer, perks in variants:
                    target_data = avg_data * buffer
                    if (not is_family and target_data > 80) or (is_family and target_data > 320):
                        data_limit = "Unlimited"
                        base_price = 149.99 if is_family else 59.99
                    else:
                        data_limit = max(step_size, round(target_data / step_size) * step_size)
                        base_price = family_base + 0.35 * data_limit if is_family else 10 + 0.40 * data_limit
                    
                    final_price = round(base_price * (1 - discount), 2)
                    full_name = f"Custom {name}{' Family' if is_family else ''} {data_limit if data_limit != 'Unlimited' else ''}{'GB' if data_limit != 'Unlimited' else 'Unlimited'}"
                    
                    features = f"• Unlimited minutes & texts\n• {perks}\n• {discount_pct}% discount ({disc_str})"
                    custom_plans.append({
                        "name": full_name,
                        "data": data_limit,
                        "price": final_price,
                        "features": features
                    })
                
                all_plans = {**standard_plans, **{p["name"]: {"data": p["data"], "price": p["price"], "features": p["features"].replace("\n", ", ")} for p in custom_plans}}
                
                prompt = f"""You are an expert telecom advisor. Analyze this customer:

{json.dumps(summary, indent=2)}

Key facts to use in justifications:
- Average data: {summary['avg_data_used_gb']}GB/month
- Over limit months: {summary['months_over_110pct']}
- Average overage cost: £{summary['avg_overage_GBP']}
- Current plan: {summary['current_plan']}
- Churn risk: {summary['churn_risk']}
- Tenure: {summary['tenure_years']} years

Available plans:
{json.dumps(all_plans, indent=2)}

Recommend EXACTLY 3 plans (mix 1-2 standard + custom). For each recommendation:

1. **Justification** (200 chars max): Data-driven reason why perfect fit
2. **Customer Message**: Exciting sales pitch mentioning their exact usage
3. **Ops Note**: Internal action items

Output ONLY valid JSON:
{{
  "top_three": ["Plan Name 1", "Plan Name 2", "Plan Name 3"],
  "justifications": ["reason1", "reason2", "reason3"],
  "customer_messages": ["msg1", "msg2", "msg3"],
  "ops_notes": ["note1", "note2", "note3"],
  "custom_plans": {json.dumps(custom_plans)}
}}"""

                try:
                    response = requests.post(
                        OPENROUTER_API_URL,
                        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                        json={
                            "model": OPENROUTER_MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.1,
                            "max_tokens": 2000
                        },
                        timeout=45
                    )
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"].strip()
                    result = json.loads(content)
                    return result
                except Exception as e:
                    return {"error": str(e)}

            if st.button("Generate AI Recommendations", type="secondary", use_container_width=True):
                with st.spinner("AI analyzing your usage..."):
                    result = get_ai_recommendations(cust_summary)
                
                if "error" in result:
                    st.error(f"AI Error: {result['error']}")
                else:
                    st.success("AI Recommendations Ready!")
                    
                    st.subheader("Custom Plans (Generated for You)")
                    custom_cols = st.columns(3)
                    for idx, cplan in enumerate(result.get("custom_plans", [])):
                        with custom_cols[idx]:
                            data_val = cplan.get("data", "Unlimited")
                            price_val = cplan.get("price", 0.0)
                            data_text = f"{data_val}GB Data" if data_val != "Unlimited" else "Unlimited Data"
                            price_text = f"£{price_val:.2f}<span style='font-size:0.7em;'>/month</span>"
                            st.markdown(f"""
                            <div class="plan-card">
                                <div class="plan-header">
                                    <span class="badge-custom">CUSTOM</span>
                                    {cplan.get('name', 'Plan')}
                                </div>
                                <div class="data-highlight">{data_text}</div>
                                <div class="price-highlight" style="font-size:1.5em;">{price_text}</div>
                                <div class="features">{cplan.get('features', '').replace('\\n', '<br>')}</div>
                                <button class="select-btn">Select This Plan</button>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.subheader("Top 3 AI Recommendations")
                    for i, plan_name in enumerate(result["top_three"]):
                        with st.container():
                            c1, c2 = st.columns([1, 3])
                            with c1:
                                plan_info = next((p for p in result.get("custom_plans", []) if p["name"] == plan_name), 
                                               standard_plans.get(plan_name, {}))
                                plan_info.setdefault("data", "Unknown")
                                plan_info.setdefault("price", 0.0)
                                data_val = plan_info["data"]
                                price_val = plan_info["price"]
                                data_text = f"{data_val}GB Data" if data_val != "Unlimited" else "Unlimited Data"
                                price_text = f"£{price_val:.2f}<span style='font-size:0.7em;'>/month</span>"
                                badge_class = "badge-custom" if "Custom" in plan_name else "badge-standard"
                                st.markdown(f"""
                                <div class="plan-card recommended">
                                    <div class="plan-header">
                                        <span class="{badge_class}">{ 'CUSTOM' if 'Custom' in plan_name else 'STANDARD'}</span>
                                        {plan_name}
                                    </div>
                                    <div class="data-highlight">{data_text}</div>
                                    <div class="price-highlight" style="font-size:1.5em;">{price_text}</div>
                                    <div class="features">{plan_info.get('features', '').replace('\\n', '<br>')}</div>
                                    <button class="select-btn">Select This Plan</button>
                                </div>
                                """, unsafe_allow_html=True)
                            with c2:
                                st.markdown(f"### #{i+1} {plan_name}")
                                st.markdown(f"**Justification:** {result['justifications'][i]}")
                                st.markdown(f'<div class="recommend-box"> <strong>Say this to customer:</strong><br>{result["customer_messages"][i]}</div>', unsafe_allow_html=True)
                                st.caption(f"**Ops:** {result['ops_notes'][i]}")

    st.markdown("---")
    st.caption(f"**Data:** `{uploaded_file1.name if uploaded_file1 else 'No file uploaded'}` | **AI:** OpenRouter GPT-3.5 | **Built for Telecom Retention**")

# ==============================
# TAB 2: BILL ANALYSIS (UPDATED)
# ==============================
with tab2:
    st.markdown("## Telecom Bill Analysis Dashboard")
    st.markdown("**AI-powered anomaly detection, bill-matching, fraud classification & valued-customer recognition**")

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

    # ---- Bulk analysis for selected customer ----
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