import streamlit as st
import anthropic
import pandas as pd
import json
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vendor Spend Optimizer",
    page_icon="💸",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 2rem; }
  .section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #111827;
    border-bottom: 2px solid #f3f4f6;
    padding-bottom: 0.4rem;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
  }
  .saving-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
  }
  .saving-card h4 { margin: 0 0 0.3rem 0; color: #166534; font-size: 0.95rem; }
  .saving-card p  { margin: 0; color: #374151; font-size: 0.88rem; }
  .issue-card {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
  }
  .issue-card h4 { margin: 0 0 0.3rem 0; color: #9a3412; font-size: 0.95rem; }
  .issue-card p  { margin: 0; color: #374151; font-size: 0.88rem; }
  .insight-card {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
  }
  .insight-card h4 { margin: 0 0 0.3rem 0; color: #1e40af; font-size: 0.95rem; }
  .insight-card p  { margin: 0; color: #374151; font-size: 0.88rem; }
  .tag {
    display: inline-block;
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.3rem;
  }
  .tag-green  { background: #dcfce7; color: #166534; }
  .tag-orange { background: #ffedd5; color: #9a3412; }
  .tag-blue   { background: #dbeafe; color: #1e40af; }
  .tag-gray   { background: #f3f4f6; color: #374151; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    st.markdown("---")
    st.markdown("**Expected CSV columns**")
    st.caption("Date, Vendor, Category, Amount")
    st.caption("Any additional columns are fine.")
    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "Upload your company's transaction history and get an AI-powered spend audit: "
        "duplicate vendors, consolidation opportunities, policy issues, and estimated savings."
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 💸 Vendor Spend Optimizer")
st.markdown(
    "Upload a transaction export and get an AI-powered spend audit — "
    "duplicate vendors, redundant SaaS, consolidation opportunities, and estimated savings."
)
st.markdown("---")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
    help="Export from your ERP, accounting tool, or corporate card platform.",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def summarize_spend(df: pd.DataFrame) -> dict:
    """Pre-compute stats to reduce tokens sent to Claude."""
    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    amount_col = next((c for c in df.columns if "amount" in c), None)
    vendor_col = next((c for c in df.columns if "vendor" in c or "merchant" in c or "description" in c), None)
    cat_col    = next((c for c in df.columns if "category" in c or "cat" in c), None)
    date_col   = next((c for c in df.columns if "date" in c), None)

    if not amount_col or not vendor_col:
        return {"raw": df.to_string(index=False)[:8000]}

    df[amount_col] = pd.to_numeric(df[amount_col].astype(str).str.replace(r"[,$]", "", regex=True), errors="coerce").fillna(0)

    vendor_summary = (
        df.groupby(vendor_col)[amount_col]
        .agg(total="sum", count="count")
        .reset_index()
        .sort_values("total", ascending=False)
    )

    stats = {
        "total_spend": round(df[amount_col].sum(), 2),
        "unique_vendors": df[vendor_col].nunique(),
        "transaction_count": len(df),
        "date_range": f"{df[date_col].min()} to {df[date_col].max()}" if date_col else "unknown",
        "top_vendors": vendor_summary.head(30).to_dict("records"),
    }

    if cat_col:
        cat_summary = df.groupby(cat_col)[amount_col].sum().reset_index().sort_values(amount_col, ascending=False)
        stats["spend_by_category"] = cat_summary.head(20).to_dict("records")

    return stats


SYSTEM_PROMPT = """You are a spend intelligence analyst at a company like Ramp or Brex.
Your job is to audit a company's vendor spend data and surface actionable savings opportunities.

Return ONLY a single valid JSON object with exactly these keys — no markdown, no code fences:

{
  "summary": {
    "total_spend_analyzed": "formatted dollar amount",
    "period": "date range as string",
    "unique_vendors": "number as string",
    "estimated_annual_savings": "dollar range e.g. $12,000–$18,000",
    "headline": "one sentence executive summary of the biggest opportunity"
  },
  "duplicate_vendors": [
    {
      "group": "e.g. AWS / Amazon Web Services / Amazon AWS",
      "vendors_found": ["list of variant names"],
      "total_spend": "formatted dollar amount",
      "action": "one sentence recommendation"
    }
  ],
  "consolidation_opportunities": [
    {
      "category": "e.g. Project Management SaaS",
      "vendors": ["Asana", "Monday.com", "Trello"],
      "combined_spend": "formatted dollar amount",
      "recommendation": "one sentence — which to keep and why",
      "estimated_saving": "dollar amount or range"
    }
  ],
  "policy_issues": [
    {
      "issue": "short label e.g. Unapproved vendor",
      "detail": "one sentence explaining the flag",
      "vendor": "vendor name if applicable",
      "amount": "dollar amount if applicable"
    }
  ],
  "spend_insights": [
    {
      "insight": "short label",
      "detail": "one sentence observation about spend patterns, trends, or anomalies"
    }
  ],
  "quick_wins": [
    "Short actionable recommendation (one per string, 3-5 total)"
  ]
}

Be specific, use the actual vendor names and numbers from the data.
Focus on what a CFO or finance team would act on immediately."""


def analyze_spend(stats: dict, api_key: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    payload = json.dumps(stats, default=str)[:10000]
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2500,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Analyze this vendor spend data and return your JSON audit:\n\n{payload}"
        }],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ── Main flow ─────────────────────────────────────────────────────────────────
if uploaded_file and api_key:
    df = pd.read_csv(uploaded_file)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows loaded", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("File", uploaded_file.name)

    with st.spinner("Summarizing spend data…"):
        stats = summarize_spend(df.copy())

    with st.spinner("Running AI spend audit…"):
        try:
            result = analyze_spend(stats, api_key)
        except json.JSONDecodeError as e:
            st.error(f"Could not parse AI response. Try again. ({e})")
            st.stop()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    # ── Summary bar ───────────────────────────────────────────────────────────
    summary = result.get("summary", {})
    st.markdown("---")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Spend Analyzed", summary.get("total_spend_analyzed", "—"))
    with s2:
        st.metric("Unique Vendors", summary.get("unique_vendors", "—"))
    with s3:
        st.metric("Period", summary.get("period", "—"))
    with s4:
        st.metric("Est. Annual Savings", summary.get("estimated_annual_savings", "—"))

    st.info(f"**Key finding:** {summary.get('headline', '')}")

    # ── Two-column layout ─────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        # Duplicate vendors
        dupes = result.get("duplicate_vendors", [])
        if dupes:
            st.markdown("<div class='section-header'>🔁 Duplicate Vendors</div>", unsafe_allow_html=True)
            for d in dupes:
                variants = ", ".join(d.get("vendors_found", []))
                st.markdown(
                    f"<div class='issue-card'>"
                    f"<h4>{d.get('group', '')} &mdash; {d.get('total_spend', '')}</h4>"
                    f"<p><span class='tag tag-orange'>Variants found</span>{variants}</p>"
                    f"<p style='margin-top:0.4rem'>{d.get('action', '')}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Policy issues
        issues = result.get("policy_issues", [])
        if issues:
            st.markdown("<div class='section-header'>🚩 Policy Issues</div>", unsafe_allow_html=True)
            for i in issues:
                vendor_str = f" &mdash; {i['vendor']}" if i.get("vendor") else ""
                amount_str = f" ({i['amount']})" if i.get("amount") else ""
                st.markdown(
                    f"<div class='issue-card'>"
                    f"<h4>{i.get('issue', '')}{vendor_str}{amount_str}</h4>"
                    f"<p>{i.get('detail', '')}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with right:
        # Consolidation opportunities
        consol = result.get("consolidation_opportunities", [])
        if consol:
            st.markdown("<div class='section-header'>🔀 Consolidation Opportunities</div>", unsafe_allow_html=True)
            for c in consol:
                vendors_str = ", ".join(c.get("vendors", []))
                saving = c.get("estimated_saving", "")
                st.markdown(
                    f"<div class='saving-card'>"
                    f"<h4>{c.get('category', '')} &mdash; {c.get('combined_spend', '')}"
                    f"{'  &nbsp;<span class=\\'tag tag-green\\'>Save ' + saving + '</span>' if saving else ''}"
                    f"</h4>"
                    f"<p><span class='tag tag-gray'>Vendors</span>{vendors_str}</p>"
                    f"<p style='margin-top:0.4rem'>{c.get('recommendation', '')}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Insights
        insights = result.get("spend_insights", [])
        if insights:
            st.markdown("<div class='section-header'>💡 Spend Insights</div>", unsafe_allow_html=True)
            for ins in insights:
                st.markdown(
                    f"<div class='insight-card'>"
                    f"<h4>{ins.get('insight', '')}</h4>"
                    f"<p>{ins.get('detail', '')}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Quick wins ────────────────────────────────────────────────────────────
    wins = result.get("quick_wins", [])
    if wins:
        st.markdown("<div class='section-header'>⚡ Quick Wins</div>", unsafe_allow_html=True)
        for w in wins:
            st.markdown(f"- {w}")

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    report_json = json.dumps(result, indent=2)
    st.download_button(
        "⬇️ Download full audit (JSON)",
        data=report_json,
        file_name="spend_audit.json",
        mime="application/json",
    )

    with st.expander("View raw spend summary sent to AI"):
        st.json(stats)

elif uploaded_file and not api_key:
    st.warning("Enter your Anthropic API key in the sidebar to run the analysis.")

else:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Upload spend data**")
        st.markdown("Export transactions from your card platform, ERP, or accounting tool as a CSV.")
    with c2:
        st.markdown("**2. AI audits your vendors**")
        st.markdown("Claude identifies duplicate vendors, redundant SaaS, consolidation plays, and policy issues.")
    with c3:
        st.markdown("**3. Act on savings**")
        st.markdown("Get a prioritized list of quick wins with estimated dollar savings your finance team can act on today.")
