# fast.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import Dict, Any
import pandas as pd
import numpy as np
import requests
import json
import io
import os

app = FastAPI(title="Telecom AI System API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# CONFIG
# ------------------------------
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-3.5-turbo"

# ------------------------------
# PLANS
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

# In-memory stores
usage_data_store = {}
summary_store = {}
analysis_done = {}
bill_data_store = {}

# ------------------------------
# SAFE JSON RESPONSE (handles NaN, inf, NaT)
# ------------------------------
def safe_json_response(data: Dict[str, Any], **kwargs):
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(i) for i in obj]
        if pd.isna(obj):  # Handles NaN, NaT, None
            return None
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj) if not (np.isnan(obj) or np.isinf(obj)) else None
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj
    return JSONResponse(content=clean(data), **kwargs)

# ------------------------------
# HELPERS
# ------------------------------
def load_and_clean_usage_data(file_content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_content))
    required = ["customer_id", "month", "plan_name", "data_used_gb", "overage_charges_GBP"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    df["month"] = pd.to_datetime(df["month"], errors='coerce')
    numeric_cols = ["data_used_gb", "overage_charges_GBP", "text_messages_sent", "call_minutes_used"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    def infer_limit(plan_name):
        s = str(plan_name)
        for p, info in standard_plans.items():
            if p in s:
                return info["data"] if info["data"] != "Unlimited" else np.inf
        return np.inf
    df["data_limit_gb"] = df["plan_name"].apply(infer_limit)
    df["tenure_years"] = pd.to_numeric(df.get("tenure_years", 1), errors='coerce').fillna(1).astype(int)
    df["satisfaction_score"] = pd.to_numeric(df.get("satisfaction_score", 8.0), errors='coerce').fillna(8.0)

    for col in ["customer_name", "uk_contact_number", "email"]:
        if col in df.columns:
            df[col] = df[col].fillna("N/A" if col != "customer_name" else "Unknown").astype(str)

    return df

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for cid, g in df.groupby("customer_id"):
        avg_data = float(g["data_used_gb"].mean())
        limit = g.iloc[-1]["data_limit_gb"]
        limit_val = np.inf if limit == np.inf else float(limit)
        months_over = int((g["data_used_gb"] > 1.1 * limit_val).sum())
        avg_overage = float(g["overage_charges_GBP"].mean())
        avg_sat = float(g["satisfaction_score"].mean())
        tenure = int(g.iloc[-1]["tenure_years"])
        plan = str(g.iloc[-1]["plan_name"])

        name = g["customer_name"].iloc[0] if "customer_name" in g.columns else "Unknown"
        phone = g["uk_contact_number"].iloc[0] if "uk_contact_number" in g.columns else "N/A"
        email = g["email"].iloc[0] if "email" in g.columns else "N/A"

        churn = "Low"
        if avg_sat < 6.0 or (months_over >= 6 and avg_overage > 2.5):
            churn = "High"
        elif avg_sat < 7.0:
            churn = "Medium"

        summaries.append({
            "customer_id": cid,
            "customer_name": name,
            "uk_contact_number": phone,
            "email": email,
            "current_plan": plan,
            "avg_data_used_gb": round(avg_data, 2),
            "data_limit_gb": "Unlimited" if limit_val == np.inf else round(limit_val, 1),
            "months_over_110pct": months_over,
            "avg_overage_GBP": round(avg_overage, 2),
            "avg_satisfaction": round(avg_sat, 1),
            "tenure_years": tenure,
            "churn_risk": churn
        })
    return pd.DataFrame(summaries)

def get_ai_recommendations(summary: Dict[str, Any]) -> Dict[str, Any]:
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
            "name": full_name.strip(),
            "data": data_limit,
            "price": final_price,
            "features": features
        })

    all_plans = {**standard_plans,
                 **{p["name"]: {"data": p["data"], "price": p["price"], "features": p["features"].replace("\n", ", ")} for p in custom_plans}}

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
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        return json.loads(content)
    except Exception as e:
        return {"error": f"AI failed: {str(e)}"}

def load_and_clean_bill_data(file_content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_content))
    rename_map = {
        'month': 'invoice_month',
        'invoice_total_GBP': 'bill_amount',
        'payment_received_GBP': 'payment_amount',
        'data_used_gb': 'data_usage_gb'
    }
    df = df.rename(columns=rename_map)
    df['dispute_flag'] = df.get('dispute_description', pd.Series()).notna() | df.get('dispute_type', pd.Series()).notna()

    numeric_cols = ["bill_amount", "payment_amount", "data_usage_gb",
                    "base_plan_charge_GBP", "overage_charges_expected_GBP",
                    "overage_charges_billed_GBP", "tax_GBP", "reconciliation_match_score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    df['dispute_flag'] = df['dispute_flag'].astype(bool)
    df['invoice_month'] = pd.to_datetime(df['invoice_month'], errors='coerce')
    df = df.sort_values(['customer_id', 'invoice_month'])
    return df

def analyze_bill_with_llm(record: Dict[str, Any]) -> Dict[str, Any]:
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
        elif txt.startswith("```"): txt = txt[3:-3].strip()
        return json.loads(txt)
    except Exception as e:
        return {"error": str(e)}

# ------------------------------
# ENDPOINTS
# ------------------------------
@app.get("/")
async def root():
    return {"message": "Telecom AI System API", "version": "1.0.0", "status": "running"}

@app.get("/api/plans/standard")
async def get_standard_plans():
    return safe_json_response({"plans": standard_plans})

@app.post("/api/usage/upload")
async def upload_usage_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(400, "Only CSV files allowed")

        content = await file.read()
        df = load_and_clean_usage_data(content)
        session_id = "default"
        usage_data_store[session_id] = df
        analysis_done[session_id] = False

        avg_plan = {k: float(v) for k, v in df.groupby('plan_name')['data_used_gb'].mean().items()}
        avg_time = df.groupby(df['month'].dt.to_period('M'))['data_used_gb'].mean()
        avg_time.index = avg_time.index.astype(str)
        avg_time = {k: float(v) for k, v in avg_time.items()}
        plan_dist = {k: int(v) for k, v in df['plan_name'].value_counts().items()}

        preview = df.head(15).replace([np.nan], [None]).to_dict(orient='records')

        return safe_json_response({
            "message": "File processed",
            "rows": len(df),
            "columns": list(df.columns),
            "preview": preview,
            "stats": {
                "avg_data_per_plan": avg_plan,
                "avg_data_over_time": avg_time,
                "plan_distribution": plan_dist
            },
            "hint": "Next: POST /api/usage/analyze"
        })
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.post("/api/usage/analyze")
async def analyze_usage():
    try:
        session_id = "default"
        if session_id not in usage_data_store:
            raise HTTPException(400, "Upload usage data first")

        df = usage_data_store[session_id]
        summary_df = compute_summary(df)
        summary_store[session_id] = summary_df
        analysis_done[session_id] = True

        return safe_json_response({
            "message": "Analysis complete",
            "total_customers": len(summary_df),
            "summaries": summary_df.to_dict(orient='records'),
            "hint": "Now call /api/recommendations/generate"
        })
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@app.post("/api/recommendations/generate")
async def generate_recommendations(customer_id: str = Form(...)):
    try:
        session_id = "default"
        if not analysis_done.get(session_id):
            raise HTTPException(400, "Run /api/usage/analyze first")

        summary_df = summary_store[session_id]
        row = summary_df[summary_df['customer_id'] == customer_id]
        if row.empty:
            raise HTTPException(404, "Customer not found")

        result = get_ai_recommendations(row.iloc[0].to_dict())
        return safe_json_response(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"AI failed: {str(e)}")

@app.get("/api/usage/customer/{customer_id}")
async def get_customer_details(customer_id: str):
    try:
        session_id = "default"
        if session_id not in usage_data_store or session_id not in summary_store:
            raise HTTPException(400, "Upload & analyze first")

        df = usage_data_store[session_id]
        summary_df = summary_store[session_id]
        s = summary_df[summary_df['customer_id'] == customer_id]
        if s.empty:
            raise HTTPException(404, "Customer not found")

        recent = df[df['customer_id'] == customer_id][['month', 'data_used_gb', 'overage_charges_GBP']].tail(6)
        recent = recent.replace([np.nan], [None]).to_dict(orient='records')

        return safe_json_response({
            "summary": s.iloc[0].to_dict(),
            "recent_usage": recent
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/usage/recent/{customer_id}")
async def get_recent_usage(customer_id: str):
    try:
        session_id = "default"
        if session_id not in usage_data_store:
            raise HTTPException(400, "Upload usage data first")

        df = usage_data_store[session_id]
        recent = df[df['customer_id'] == customer_id][['month', 'data_used_gb', 'overage_charges_GBP']].tail(6)
        if recent.empty:
            raise HTTPException(404, "No recent usage found")

        recent = recent.replace([np.nan], [None]).to_dict(orient='records')
        return safe_json_response({"recent_usage": recent})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/bills/upload")
async def upload_bill_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(400, "Only CSV allowed")

        content = await file.read()
        df = load_and_clean_bill_data(content)
        session_id = "default"
        bill_data_store[session_id] = df

        total_customers = len(df['customer_id'].unique())

        anom_cust = set(df[df.get('billing_anomaly_flag', False) == True]['customer_id']) if 'billing_anomaly_flag' in df.columns else set()
        num_anom_cust = len(anom_cust)

        mis_bills = df[df.get('reconciliation_match_score', 1) < 0.8][['customer_id', 'invoice_month', 'reconciliation_match_score']] if 'reconciliation_match_score' in df.columns else pd.DataFrame(columns=['customer_id', 'invoice_month', 'reconciliation_match_score'])
        num_mis_bills = len(mis_bills)
        mis_cust = set(mis_bills['customer_id'])

        disp_bills = df[df['dispute_flag'] == True][['customer_id', 'invoice_month', 'dispute_type']]
        num_disp_bills = len(disp_bills)
        disp_cust = set(disp_bills['customer_id'])

        bad = anom_cust | mis_cust | disp_cust
        valued_cust = [c for c in df['customer_id'].unique() if c not in bad]
        num_valued_cust = len(valued_cust)

        # Additional aggregations for charts
        monthly_bills = df.groupby(df['invoice_month'].dt.to_period('M')).agg({
            'bill_amount': 'sum',
            'customer_id': 'nunique',
            'dispute_flag': 'sum'
        }).reset_index()
        monthly_bills['invoice_month'] = monthly_bills['invoice_month'].dt.to_timestamp()

        bill_dist = pd.cut(df['bill_amount'], bins=[0, 50, 100, 200, 500, 1000, float('inf')], 
                           labels=['£0-50', '£50-100', '£100-200', '£200-500', '£500-1000', '£1000+']).value_counts().reset_index()
        bill_dist.columns = ['Bill Range', 'Count']
        bill_dist_dict = {row['Bill Range']: int(row['Count']) for _, row in bill_dist.iterrows()}

        monthly_dispute_status = df.groupby([df['invoice_month'].dt.to_period('M'), 'dispute_flag']).size().unstack(fill_value=0).reset_index()
        monthly_dispute_status.columns = ['Invoice Month', 'No Dispute', 'Dispute']
        monthly_dispute_status['Invoice Month'] = monthly_dispute_status['Invoice Month'].dt.to_timestamp()
        if 'No Dispute' not in monthly_dispute_status.columns:
            monthly_dispute_status['No Dispute'] = 0
        if 'Dispute' not in monthly_dispute_status.columns:
            monthly_dispute_status['Dispute'] = 0
        dispute_trend = {row['Invoice Month'].strftime('%Y-%m'): {'no_dispute': int(row['No Dispute']), 'dispute': int(row['Dispute'])} for _, row in monthly_dispute_status.iterrows()}

        status_counts = {
            'Anomalous': num_anom_cust,
            'Mismatched': len(mis_cust),
            'Disputed': len(disp_cust),
            'Valued': num_valued_cust
        }

        status_bar = status_counts

        preview = df.head(10).replace([np.nan], [None]).to_dict(orient='records')
        customers = sorted(df['customer_id'].unique().tolist())
        months = sorted(df['invoice_month'].dt.strftime('%Y-%m').dropna().unique().tolist())

        return safe_json_response({
            "message": "Bill file processed",
            "rows": len(df),
            "statistics": {
                "total_customers": total_customers,
                "anomalous_customers": num_anom_cust,
                "mismatched_bills": num_mis_bills,
                "disputed_bills": num_disp_bills,
                "valued_customers": num_valued_cust
            },
            "charts": {
                "status_counts": status_counts,
                "bill_distribution": bill_dist_dict,
                "dispute_trend": dispute_trend,
                "status_bar": status_bar
            },
            "preview": preview,
            "customers": customers,
            "months": months
        })
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/bills/analyze")
async def analyze_bill(customer_id: str = Form(...), month: str = Form(...)):
    try:
        session_id = "default"
        if session_id not in bill_data_store:
            raise HTTPException(400, "Upload bill data first")

        df = bill_data_store[session_id]
        rec = df[(df['customer_id'] == customer_id) & (df['invoice_month'].dt.strftime('%Y-%m') == month)]
        if rec.empty:
            raise HTTPException(404, "Bill not found")

        result = analyze_bill_with_llm(rec.iloc[0].to_dict())
        return safe_json_response(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/bills/bulk_analyze")
async def bulk_analyze_bills(customer_id: str = Form(...)):
    try:
        session_id = "default"
        if session_id not in bill_data_store:
            raise HTTPException(400, "Upload bill data first")

        df = bill_data_store[session_id]
        sub = df[df['customer_id'] == customer_id]
        if sub.empty:
            raise HTTPException(404, "No bills found for customer")

        out = []
        for _, r in sub.iterrows():
            res = analyze_bill_with_llm(r.to_dict())
            flat = {
                'anomaly_score': res.get('anomaly_detection', {}).get('score', 0.0),
                'match_score': res.get('matching_model', {}).get('match_score', 0.0),
                'fraud_prob': res.get('fraud_dispute_classifier', {}).get('fraud_probability', 0.0),
            }
            bill_info = r.to_dict()
            out.append({**bill_info, **flat})

        # Compute bulk stats
        bulk_df = pd.DataFrame(out)
        total_bills = len(bulk_df)
        total_bill_amount = float(bulk_df['bill_amount'].sum())
        avg_anomaly = float(bulk_df['anomaly_score'].mean())
        avg_match = float(bulk_df['match_score'].mean())
        avg_fraud = float(bulk_df['fraud_prob'].mean())

        return safe_json_response({
            "message": "Bulk analysis complete",
            "bulk_results": out,
            "stats": {
                "total_bills": total_bills,
                "total_bill_amount": total_bill_amount,
                "avg_anomaly": avg_anomaly,
                "avg_match": avg_match,
                "avg_fraud": avg_fraud
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/bills/categories")
async def get_bill_categories():
    try:
        session_id = "default"
        if session_id not in bill_data_store:
            raise HTTPException(400, "Upload bill data first")

        df = bill_data_store[session_id]
        anom = df[df.get('billing_anomaly_flag', False) == True][['customer_id', 'customer_name', 'uk_contact_number', 'email']].drop_duplicates().to_dict(orient='records')
        mis = df[df.get('reconciliation_match_score', 1) < 0.8][['customer_id', 'invoice_month', 'reconciliation_match_score', 'customer_name', 'uk_contact_number', 'email']].drop_duplicates().to_dict(orient='records')
        disp = df[df['dispute_flag'] == True][['customer_id', 'invoice_month', 'dispute_type', 'customer_name', 'uk_contact_number', 'email']].drop_duplicates().to_dict(orient='records')

        bad = set([item['customer_id'] for item in anom] + [item['customer_id'] for item in mis] + [item['customer_id'] for item in disp])
        valued = df[~df['customer_id'].isin(bad)][['customer_id', 'customer_name', 'uk_contact_number', 'email']].drop_duplicates().to_dict(orient='records')

        return safe_json_response({
            "categories": {
                "anomalous": anom,
                "mismatched": mis,
                "disputed": disp,
                "valued": valued
            }
        })
    except Exception as e:
        raise HTTPException(500, str(e))

# Static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/index.html")
async def serve_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "index.html not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)