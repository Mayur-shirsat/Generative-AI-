import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px

# ------------------------------
# CONFIG & STYLE
# ------------------------------
st.set_page_config(page_title="Smart Telecom Plan Recommender", layout="wide")

# Enhanced CSS for professional plan cards
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
</style>
""", unsafe_allow_html=True)

# API Config
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
# DATA LOADING & CLEANING
# ------------------------------
@st.cache_data
def load_and_clean_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    required_cols = ["customer_id", "month", "plan_name", "data_used_gb", "overage_charges_GBP"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    
    # Convert month to datetime
    df["month"] = pd.to_datetime(df["month"], errors='coerce')
    
    # Type conversions
    numeric_cols = ["data_used_gb", "overage_charges_GBP", "text_messages_sent", "call_minutes_used"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Infer data limits from plan names
    def infer_limit(plan_name):
        for plan, info in standard_plans.items():
            if plan in str(plan_name):
                return info["data"] if info["data"] != "Unlimited" else np.inf
        return np.inf  # Default unlimited
    
    df["data_limit_gb"] = df["plan_name"].apply(infer_limit)
    
    # Defaults
    df["tenure_years"] = pd.to_numeric(df.get("tenure_years", 1), errors='coerce').fillna(1).astype(int)
    df["satisfaction_score"] = pd.to_numeric(df.get("satisfaction_score", 8.0), errors='coerce').fillna(8.0)
    
    # Ensure contact info columns are clean
    if "customer_name" in df.columns:
        df["customer_name"] = df["customer_name"].fillna("Unknown").astype(str)
    if "uk_contact_number" in df.columns:
        df["uk_contact_number"] = df["uk_contact_number"].fillna("N/A").astype(str)
    if "email" in df.columns:
        df["email"] = df["email"].fillna("N/A").astype(str)
    
    return df

# ------------------------------
# CUSTOMER SUMMARY
# ------------------------------
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
        
        # Get contact info (from any row, assuming consistent per customer)
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

# ------------------------------
# AI RECOMMENDATION ENGINE
# ------------------------------
def get_ai_recommendations(summary):
    # Calculate personalized discount
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
    
    # Generate 3 custom plans
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
    
    # All available plans for AI context
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

# ------------------------------
# PLAN CARD RENDERER
# ------------------------------
def render_plan_card(plan_info, is_recommended=False):
    data_val = plan_info.get("data", "Unlimited")
    price_val = plan_info.get("price", 0.0)
    
    data_text = f"{data_val}GB Data" if data_val != "Unlimited" else "Unlimited Data"
    price_text = f"£{price_val:.2f}<span style='font-size:0.7em;'>/month</span>"
    badge_class = "badge-custom" if "Custom" in plan_info.get('name', '') else "badge-standard"
    
    return f"""
    <div class="plan-card{' recommended' if is_recommended else ''}">
        <div class="plan-header">
            <span class="{badge_class}">{ 'CUSTOM' if 'Custom' in plan_info.get('name', '') else 'STANDARD'}</span>
            {plan_info.get('name', 'Plan')}
        </div>
        <div class="data-highlight">{data_text}</div>
        <div class="price-highlight" style="font-size:1.5em;">{price_text}</div>
        <div class="features">{plan_info.get('features', '').replace('\\n', '<br>')}</div>
        <button class="select-btn">Select This Plan</button>
    </div>
    """

# ------------------------------
# MAIN UI
# ------------------------------
st.title("Smart Telecom Plan Recommender")
st.markdown("**AI analyzes your usage patterns and recommends perfect plans with data-backed justifications**")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df_monthly = load_and_clean_data(uploaded_file)

    # Data Preview
    with st.expander("Data Preview", expanded=True):
        display_df = df_monthly.head(15).copy()
        display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
        
        format_dict = {
            'Data Used Gb': '{:.1f}',
            'Overage Charges Gbp': '£{:.2f}',
            'Base Plan Charge Gbp': '£{:.2f}',
            'Invoice Total Gbp': '£{:.2f}'
        }
        st.dataframe(
            display_df.style.format(format_dict).background_gradient(cmap='viridis', subset=['Data Used Gb']),
            height=400,
            use_container_width=True
        )
        
        # Add visualizations
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

    # Compute summaries
    if st.button("Analyze All Customers", type="primary", use_container_width=True):
        with st.spinner("Computing usage summaries..."):
            df_summary = compute_summary(df_monthly)
            st.session_state.df_summary = df_summary
            st.session_state.df_monthly = df_monthly  # Save full data for contact lookup
            st.success("Analysis complete!")
        
        st.dataframe(df_summary.head(10), use_container_width=True)
        st.download_button("Download Summaries", df_summary.to_csv(index=False).encode(), "customer_analysis.csv")

    if "df_summary" in st.session_state:
        df_summary = st.session_state.df_summary
        df_monthly = st.session_state.df_monthly
        
        # Standard Plans Grid
        st.subheader("Standard Plans")
        cols = st.columns(3)
        for idx, (name, info) in enumerate(standard_plans.items()):
            with cols[idx % 3]:
                st.markdown(render_plan_card({"name": name, **info}), unsafe_allow_html=True)
        
        # Personalized Recommender
        st.subheader("Your Personalized Recommendations")
        customer = st.selectbox("Select Customer", df_summary["customer_id"].tolist(), key="customer_select")
        cust_summary = df_summary[df_summary["customer_id"] == customer].iloc[0].to_dict()
        
        # Extract contact info from full dataset
        contact_row = df_monthly[df_monthly["customer_id"] == customer].iloc[0]
        customer_name = contact_row.get("customer_name", "Unknown")
        uk_contact_number = contact_row.get("uk_contact_number", "N/A")
        email = contact_row.get("email", "N/A")
        
        # Customer Profile in Attractive Box
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

        if st.button("Generate AI Recommendations", type="secondary", use_container_width=True):
            with st.spinner("AI analyzing your usage..."):
                result = get_ai_recommendations(cust_summary)
            
            if "error" in result:
                st.error(f"AI Error: {result['error']}")
            else:
                st.success("AI Recommendations Ready!")
                
                # Custom Plans
                st.subheader("Custom Plans (Generated for You)")
                custom_cols = st.columns(3)
                for idx, cplan in enumerate(result.get("custom_plans", [])):
                    with custom_cols[idx]:
                        st.markdown(render_plan_card(cplan), unsafe_allow_html=True)
                
                # Top 3 Recommendations
                st.subheader("Top 3 AI Recommendations")
                for i, plan_name in enumerate(result["top_three"]):
                    with st.container():
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            plan_info = next((p for p in result.get("custom_plans", []) if p["name"] == plan_name), 
                                           standard_plans.get(plan_name, {}))
                            if "data" not in plan_info:
                                plan_info["data"] = "Unknown"
                            if "price" not in plan_info:
                                plan_info["price"] = 0.0
                            st.markdown(render_plan_card({"name": plan_name, **plan_info}, is_recommended=True), 
                                      unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"### #{i+1} {plan_name}")
                            st.markdown(f"**Justification:** {result['justifications'][i]}")
                            st.markdown(f'<div class="recommend-box"> <strong>Say this to customer:</strong><br>{result["customer_messages"][i]}</div>', unsafe_allow_html=True)
                            st.caption(f"**Ops:** {result['ops_notes'][i]}")

st.markdown("---")
st.caption(f"**Data:** `{uploaded_file.name if uploaded_file else 'No file uploaded'}` | **AI:** OpenRouter GPT-3.5 | **Built for Telecom Retention**")