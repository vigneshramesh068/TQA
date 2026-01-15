
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import html
from datetime import datetime
import plotly.express as px
import io

# =========================
# Page & Title
# =========================
st.set_page_config(page_title="Ticket Quality Audit (Rule-based)", layout="wide")
st.title("🧮 Ticket Quality Audit — Rule-based (no Transformer)")
st.markdown("---")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("📁 Upload Files")
    ticket_file = st.file_uploader("Upload Ticket Data (Excel/CSV)", type=['xlsx', 'xls', 'csv'])
    mapping_file = st.file_uploader("Upload Mapping File (Excel/CSV) — optional", type=['xlsx', 'xls', 'csv'])
    run_audit = st.button("🚀 Run Audit", type="primary", use_container_width=True)

# =========================
# Column map (case-insensitive → internal)
# =========================
COLUMN_MAP_LOWER = {
    "number": "number",
    "opened": "created_date",
    "resolved": "resolved_date",
    "closed": "closed_date",
    "category": "category",
    "work notes": "work_notes",
    "additional comments": "additional_comments",
    "resolution notes": "resolution_summary",
    "resolution summary": "resolution_summary",
    "closure notes": "resolution_summary",
    "closure summary": "resolution_summary",
    "assignment group": "assignment_group",
    "service offering": "service_offering",
    "reopen count": "reopen",
    "reassignment count": "reassignment",
    "assigned to": "assigned_to",
    "portfolio": "portfolio",
    "domain": "domain",
}

# =========================
# Helpers
# =========================
def load_file(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename headers using case-insensitive mapping and parse dates."""
    df = df.copy()
    lower_to_orig = {str(c).strip().lower(): c for c in df.columns}
    rename_dict = {}
    for k_lower, v_internal in COLUMN_MAP_LOWER.items():
        if k_lower in lower_to_orig:
            rename_dict[lower_to_orig[k_lower]] = v_internal
    df = df.rename(columns=rename_dict)

    # Parse date columns if present
    for dcol in ['created_date', 'resolved_date', 'closed_date']:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors='coerce')

    return df

def calculate_business_days(start_date, end_date):
    try:
        if pd.isna(start_date) or pd.isna(end_date):
            return None
        return int(np.busday_count(pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()))
    except Exception:
        return None

def check_breach(days, category):
    """SLA: Bug-like categories >10 business days else >3 business days."""
    if pd.isna(days):
        return None
    cat = str(category).lower()
    return days > 10 if ('bug' in cat or 'defect' in cat) else days > 3

def detect_3sr_policy(text: str) -> bool:
    """
    Heuristic for 3SR (3 reminders + no response) or explicit policy wording.
    Triggers if:
      - >= 3 reminders OR phrases like "3rd reminder" / "final reminder" / "3-strike" / "3SR"
      - AND phrases like "no response / no reply / unresponsive / no update from user"
      - OR explicit closure wording "closed as per 3-strike rule / 3SR"
    """
    if not text:
        return False

    s = text.lower()
    # Decode HTML and normalize
    s = html.unescape(s)
    s = re.sub(r'<[^>]+>', ' ', s)          # strip basic HTML tags
    s = s.replace('follow-up', 'follow up')  # unify
    s = re.sub(r'\s+', ' ', s).strip()

    # Count reminders (generic + ordinal)
    reminder_count = s.count("reminder")
    reminder_count += len(re.findall(r'\b(1st|2nd|3rd|first|second|third)\s+reminder\b', s))

    strong_markers = any(k in s for k in [
        '3sr', 'third reminder', '3rd reminder', 'final reminder',
        '3-strike', '3 strike', 'three strike'
    ])
    explicit_policy = any(k in s for k in [
        'closed as per 3-strike rule', 'closed as per three strike rule', 'closed as per 3sr'
    ])
    no_response = any(k in s for k in [
        'no response', 'no reply', 'not responded', 'unresponsive',
        'no updates from user', 'no update from user', "didn't get any response"
    ])

    return (explicit_policy or ((reminder_count >= 3) or strong_markers) and no_response)

def check_notes_quality(notes, comments, resolution_summary=None):
    """
    Rule-based scoring (max 4 here):
      - Notes exist & detailed: +2 (exists + detail)
      - User communication: +1
      - User confirmation OR 3SR: +1

    Resolution (+1) & Mapping (+1) are scored by caller.

    IMPORTANT:
    - If User Confirmed OR 3SR closure is detected, treat communication as present (no penalty)
      and never show a contradictory "No evidence of user communication".
    - Flags "regular follow-ups" when ≥2 follow-up events AND ≥2 distinct dates.
    """
    score = 0
    issues = []

    # --- Normalize & combine text ---
    parts = []
    for t in (notes, comments, resolution_summary):
        t = '' if pd.isna(t) else str(t)
        parts.append(t)
    raw = "\n".join(parts)

    # Decode HTML entities, strip tags, unify variants
    s = html.unescape(raw)
    s = re.sub(r'<[^>]+>', ' ', s)          # remove tags like <...>
    s = s.replace('&nbsp;', ' ')
    s = s.replace('follow-up', 'follow up') # unify hyphenation
    s = s.replace('Follow-up', 'follow up')
    s = re.sub(r'[–—]', '-', s)             # long dashes → hyphen
    s = re.sub(r'\s+', ' ', s).strip()
    s_lower = s.lower()

    # --- Basic presence ---
    if len(s_lower) < 10:
        issues.append("Empty or minimal documentation")
        return score, issues, {
            'has_user_update': False,
            'has_confirmation': False,
            'closed_by_3sr': False,
            'closure_policy': '',
            'regular_followups': False
        }

    # +1 for having content
    score += 1

    # --- Phrase banks ---
    user_update_phrases = [
        'informed user', 'updated customer', 'notified user', 'contacted user',
        'emailed user', 'followed up', 'follow up', 'reminder sent', 'sent reminder',
        'called user', 'teams message', 'pinged user',
        'tried to connect', 'attempted to contact', 'attempted to reach', 'reached out',
        'tried contacting', 'tried to contact', 'connected with user'
    ]

    # Expanded confirmation patterns (regex, case-insensitive)
    confirmation_regexes = [
        r'\b(user|customer)\s+(confirmed|approval|approved|acknowledged)\b',
        r'\bconfirmation\s+(received|acknowledged)\b',
        r'\bthanks?\s+for\s+the\s+confirmation\b',
        r'\b(ok(ay)?\s+to\s+close|please\s+close\s+(the\s+)?ticket|ticket\s+can\s+be\s+closed)\b',
        r'\b(view|report|form|screen|process)\s+is\s+working\b',
        r'\b(working\s+fine\s+now|working\s+as\s+expected)\b',
        r'\b(fixed\s+and\s+verified|issue\s+resolved|problem\s+resolved|resolution\s+accepted)\b',
        r'\bverified\s+by\s+(user|customer)\b',
        r'\bthank(s| you)[\s,]+.*(it|this)\s+works\b',
        r'\bcan\s+be\s+closed\b',
    ]

    # Regex to detect dates like 2025-10-27 / 2025/10/27 / 10/27/2025
    date_hits = set(re.findall(r'(20\d{2}[-/]\d{2}[-/]\d{2}|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b)', s_lower))

    # Count follow-up markers (regularity heuristic)
    followup_markers = [
        'reminder', 'follow up', 'contacted', 'emailed', 'called',
        'reached out', 'tried to connect', 'attempted to'
    ]
    followup_events = sum(s_lower.count(m) for m in followup_markers)

    # --- Detail scoring ---
    detailed = (followup_events >= 2 and len(date_hits) >= 2) or (len(s_lower) > 200)
    if detailed:
        score += 1
    else:
        if followup_events < 2:
            issues.append("Insufficient updates")
        if len(s_lower) <= 200:
            issues.append("Updates lack detail")

    # --- Detect confirmation and 3SR first (so they imply communication) ---
    has_confirmation = any(re.search(rx, s_lower) for rx in confirmation_regexes)
    closed_by_3sr = detect_3sr_policy(s_lower)
    closure_policy = 'User Confirmed' if has_confirmation else ('3SR' if closed_by_3sr else '')

    # --- Communication detection (with implication) ---
    has_user_update = any(p in s_lower for p in user_update_phrases)

    # If we have User Confirmed OR 3SR, treat communication as present (no penalty)
    if has_confirmation or closed_by_3sr:
        has_user_update = True

    if has_user_update:
        score += 1
    else:
        # softer signals (still no confirmation/3SR)
        soft_signals = any(m in s_lower for m in followup_markers)
        if soft_signals:
            score += 1
        else:
            issues.append("No evidence of user communication")

    # --- Confirmation OR 3SR credit ---
    if has_confirmation:
        score += 1
    elif closed_by_3sr:
        score += 1
        # informational note; not a penalty
        issues.append("Closed via 3SR policy (no user response after follow-ups)")
    else:
        issues.append("No user confirmation documented")

    # Deduplicate issues & build info
    issues = list(dict.fromkeys(issues))
    info = {
        'has_user_update': has_user_update,
        'has_confirmation': has_confirmation,
        'closed_by_3sr': closed_by_3sr,
        'closure_policy': closure_policy,
        'regular_followups': (followup_events >= 2 and len(date_hits) >= 2)
    }
    return score, issues, info

def check_resolution_quality(resolution):
    if pd.isna(resolution) or len(str(resolution).strip()) < 20:
        return False, "Resolution summary missing or too brief"
    return True, "Good resolution summary"

# --------- 1:N AG→SO mapping ----------
def build_mapping_lookup(mapping_df: pd.DataFrame):
    """
    Build a case-insensitive lookup:
      ag_lower -> set([so_lower, ...]) and a display list for messages.
    Expects mapping_df with first two columns: Assignment Group, Service Offering
    """
    if mapping_df is None or mapping_df.empty:
        return {}, {}

    # Use only first two columns
    m = mapping_df.iloc[:, :2].copy()
    # Normalize to strings; strip
    m.iloc[:, 0] = m.iloc[:, 0].astype(str).str.strip()
    m.iloc[:, 1] = m.iloc[:, 1].astype(str).str.strip()

    lookup = {}
    display = {}

    for _, r in m.iterrows():
        ag_raw = r.iloc[0]
        so_raw = r.iloc[1]
        if not ag_raw or str(ag_raw).lower() in ['nan', 'none']:
            continue
        ag_l = str(ag_raw).strip().lower()
        so_l = str(so_raw).strip().lower() if pd.notna(so_raw) else ''

        # Initialize containers
        if ag_l not in lookup:
            lookup[ag_l] = set()
            display[ag_l] = set()
        # Add only non-empty SOs
        if so_l and so_l not in ['nan', 'none']:
            lookup[ag_l].add(so_l)
            display[ag_l].add(str(so_raw).strip())

    # Convert display to sorted list for stable messages
    display = {k: sorted(list(v)) for k, v in display.items()}
    return lookup, display

def validate_mapping_multi(assignment_group, service_offering, mapping_lookup, mapping_display):
    """
    Validate against 1:N mapping. Returns (bool, message).
    """
    ag_str = '' if pd.isna(assignment_group) else str(assignment_group).strip()
    so_str = '' if pd.isna(service_offering) else str(service_offering).strip()

    ag_l = ag_str.lower()
    so_l = so_str.lower()

    if ag_l not in mapping_lookup:
        return False, "Assignment group not found in mapping"

    allowed_set = mapping_lookup.get(ag_l, set())
    allowed_list = mapping_display.get(ag_l, [])

    if not so_l:
        return False, f"Service offering missing for '{ag_str}'. Expected one of: {', '.join(allowed_list) if allowed_list else '(none listed)'}"

    if so_l in allowed_set:
        return True, "Correct mapping"
    else:
        return False, f"Incorrect service offering for '{ag_str}' (got: {so_str}). Expected one of: {', '.join(allowed_list) if allowed_list else '(none listed)'}"
# --------------------------------------

def month_frame_from_closed(df: pd.DataFrame, closed_col='Closed_Date'):
    """Return df with MonthPeriod & MonthLabel from Closed_Date (MMM YYYY)."""
    tmp = df.copy()
    if closed_col not in tmp.columns:
        return pd.DataFrame()
    tmp[closed_col] = pd.to_datetime(tmp[closed_col], errors='coerce')
    tmp = tmp.dropna(subset=[closed_col])
    if tmp.empty:
        return tmp
    tmp['MonthPeriod'] = tmp[closed_col].dt.to_period('M')
    tmp['MonthLabel'] = tmp['MonthPeriod'].dt.strftime('%b %Y')
    return tmp

def color_map_for(categories):
    palette = px.colors.qualitative.Plotly
    mapping = {}
    for i, c in enumerate(categories):
        mapping[c] = palette[i % len(palette)]
    return mapping

def build_sla_100_stack(df: pd.DataFrame, entity_col: str, title: str):
    """
    Build a 100%-stacked bar chart for SLA outcomes WITHOUT barnorm/%{percent}.
    We compute Percent per entity manually and plot Percent as y; label with Percent and raw Count.
    Required columns: entity_col, Ticket_Count, Breach_Count
    """
    tmp = df[[entity_col, 'Ticket_Count', 'Breach_Count']].copy()
    tmp['SLA_Met'] = tmp['Ticket_Count'] - tmp['Breach_Count']
    tmp['Den'] = tmp['Ticket_Count'].replace(0, np.nan)

    stack = tmp.melt(
        id_vars=[entity_col, 'Ticket_Count', 'Den'],
        value_vars=['SLA_Met', 'Breach_Count'],
        var_name='Outcome', value_name='Count'
    )
    stack['Percent'] = np.where(stack['Den'].notna(), (stack['Count'] / stack['Den']) * 100, 0)
    stack['Outcome'] = stack['Outcome'].map({'SLA_Met': 'SLA Met', 'Breach_Count': 'Breached'})

    fig = px.bar(
        stack, x=entity_col, y='Percent',
        color='Outcome', barmode='stack',
        title=title,
        text='Percent',
        color_discrete_map={'SLA Met': '#2ca02c', 'Breached': '#d62728'},
        custom_data=['Count']
    )
    fig.update_traces(
        texttemplate='%{text:.0f}%',
        hovertemplate='<b>%{x}</b><br>%{color}: %{y:.1f}% (%{customdata[0]:.0f})<extra></extra>'
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(title='Percentage', range=[0, 100])
    return fig

# =========================
# Main
# =========================
if run_audit:
    if not ticket_file:
        st.error("❌ Please upload ticket data file")
    else:
        with st.spinner("🔄 Processing tickets..."):
            try:
                # Load files
                tickets_df = load_file(ticket_file)
                tickets_df = normalize_columns(tickets_df)

                mapping_df = None
                mapping_lookup = {}
                mapping_display = {}
                if mapping_file:
                    mapping_df = load_file(mapping_file)
                    # Build 1:N lookup for mapping
                    mapping_lookup, mapping_display = build_mapping_lookup(mapping_df)

                st.success(f"✅ Loaded {len(tickets_df)} tickets")

                # Quality max score depends on mapping presence
                MAX_SCORE = 6 if mapping_df is not None else 5

                # Process tickets
                results = []
                progress = st.progress(0)
                for idx, row in tickets_df.iterrows():
                    progress.progress((idx + 1) / len(tickets_df))

                    res = {
                        'Ticket_ID': idx + 1,
                        'Ticket_Number': row.get('number', ''),
                        'Assigned_To': row.get('assigned_to', ''),
                        'Assignment_Group': row.get('assignment_group', ''),
                        'Portfolio': row.get('portfolio', ''),
                        'Domain': row.get('domain', ''),
                        'Closed_Date': row.get('closed_date', ''),
                        'Category': row.get('category', ''),
                        'Duration_Days': None,
                        'Breach': 0,
                        'Reopen_Count': 0,
                        'Reassignment_Count': 0,
                        'Issues': [],
                        'Closure_Policy': '',
                        'Quality_Score': 0,          # out of MAX_SCORE
                        'Quality_Max': MAX_SCORE,
                    }

                    # SLA calc
                    if ('created_date' in tickets_df.columns) and ('resolved_date' in tickets_df.columns):
                        days = calculate_business_days(row.get('created_date'), row.get('resolved_date'))
                        res['Duration_Days'] = days
                        is_breach = check_breach(days, row.get('category', ''))
                        res['Breach'] = 1 if (is_breach is True) else 0
                        if is_breach:
                            res['Issues'].append(f"SLA breach ({days} business days)")

                    # Notes quality (+ communication + confirmation/3SR)
                    nq_score, nq_issues, info = check_notes_quality(
                        row.get('work_notes'),
                        row.get('additional_comments'),
                        row.get('resolution_summary') if 'resolution_summary' in tickets_df.columns else None
                    )
                    res['Quality_Score'] += nq_score
                    res['Issues'].extend(nq_issues)
                    res['Closure_Policy'] = info.get('closure_policy', '')

                    # Resolution summary credit
                    if 'resolution_summary' in tickets_df.columns:
                        has_res, res_msg = check_resolution_quality(row.get('resolution_summary'))
                        if has_res:
                            res['Quality_Score'] += 1
                        else:
                            res['Issues'].append(res_msg)

                    # Mapping credit (only if file provided) — 1:N validation
                    if mapping_df is not None:
                        ok, msg = validate_mapping_multi(
                            row.get('assignment_group'),
                            row.get('service_offering'),
                            mapping_lookup,
                            mapping_display
                        )
                        if ok:
                            res['Quality_Score'] += 1
                        else:
                            res['Issues'].append(msg)

                    # Reopen/Reassignment
                    try:
                        rc = int(float(row.get('reopen', 0)))
                        res['Reopen_Count'] = rc
                        if rc > 0:
                            res['Issues'].append(f"Ticket reopened {rc} time(s)")
                    except Exception:
                        pass

                    try:
                        rac = int(float(row.get('reassignment', 0)))
                        res['Reassignment_Count'] = rac
                        if rac > 0:
                            res['Issues'].append(f"Ticket reassigned {rac} time(s)")
                    except Exception:
                        pass

                    results.append(res)

                progress.empty()
                results_df = pd.DataFrame(results)

                # Quality %
                results_df['Quality_Percent'] = (results_df['Quality_Score'] / results_df['Quality_Max']) * 100.0

                # KPIs
                st.markdown("---")
                st.header("📊 Audit Summary")
                total_tickets = len(results_df)
                breached = int(results_df['Breach'].sum()) if total_tickets else 0
                sla_met_pct = (1 - breached / total_tickets) * 100 if total_tickets else 0
                avg_quality = results_df['Quality_Score'].mean() if total_tickets else 0
                avg_quality_pct = results_df['Quality_Percent'].mean() if total_tickets else 0
                total_reopens = int(results_df['Reopen_Count'].sum())

                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.metric("Total Tickets", f"{total_tickets:,}")
                with c2:
                    st.metric("Breached Tickets", f"{breached:,}")
                with c3:
                    st.metric(f"Avg Quality Score (/ {int(results_df['Quality_Max'].max() if total_tickets else MAX_SCORE)})", f"{avg_quality:.1f}/{int(results_df['Quality_Max'].max() if total_tickets else MAX_SCORE)}")
                with c4:
                    st.metric("Avg Quality %", f"{avg_quality_pct:.1f}%")
                with c5:
                    st.metric("SLA Met %", f"{sla_met_pct:.1f}%")

                # ========= Month-based (Closed → MMM YYYY) =========
                st.markdown("---")
                st.header("📈 Trend & Performance Analysis")

                st.subheader("📆 Month-over-Month (Closed Month)")
                mdf = month_frame_from_closed(results_df, closed_col='Closed_Date')
                if not mdf.empty:
                    monthly = mdf.groupby(['MonthPeriod', 'MonthLabel']).agg(
                        Ticket_Count=('Ticket_ID', 'count'),
                        Breach_Count=('Breach', 'sum'),
                        Avg_Duration=('Duration_Days', 'mean')
                    ).reset_index().sort_values('MonthPeriod')
                    monthly['Breach_Rate'] = (monthly['Breach_Count'] / monthly['Ticket_Count']) * 100

                    # Volume (integers)
                    fig_m1 = px.bar(
                        monthly, x='MonthLabel', y='Ticket_Count',
                        title='Monthly Ticket Volume (Closed Month)',
                        text='Ticket_Count',
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_m1.update_traces(texttemplate='%{text:.0f}', hovertemplate='<b>%{x}</b><br>Tickets: %{y:.0f}<extra></extra>')
                    fig_m1.update_xaxes(title=None)
                    fig_m1.update_yaxes(title='Ticket Count', tickformat=',d')
                    st.plotly_chart(fig_m1, use_container_width=True)

                    # Breach Rate
                    fig_m2 = px.line(
                        monthly, x='MonthLabel', y='Breach_Rate',
                        title='Breach Rate (%) by Month',
                        markers=True
                    )
                    fig_m2.update_traces(line_color='#EF553B', hovertemplate='<b>%{x}</b><br>Breach Rate: %{y:.1f}%<extra></extra>')
                    fig_m2.update_xaxes(title=None)
                    fig_m2.update_yaxes(title='Breach Rate (%)', range=[0, 100])
                    st.plotly_chart(fig_m2, use_container_width=True)

                    # Avg Duration
                    fig_m3 = px.bar(
                        monthly, x='MonthLabel', y='Avg_Duration',
                        title='Average Duration (Business Days) by Month',
                        text='Avg_Duration',
                        color_discrete_sequence=['#9467bd']
                    )
                    fig_m3.update_traces(texttemplate='%{text:.1f}', hovertemplate='<b>%{x}</b><br>Avg Duration: %{y:.1f} days<extra></extra>')
                    fig_m3.update_xaxes(title=None)
                    fig_m3.update_yaxes(title='Avg Duration (days)')
                    st.plotly_chart(fig_m3, use_container_width=True)
                else:
                    st.info("ℹ️ No valid Closed dates to build month-over-month charts.")

                # ========= Assigned-To =========
                st.subheader("👤 Assigned-To Performance")
                if 'Assigned_To' in results_df.columns:
                    at_agg = results_df.groupby('Assigned_To').agg(
                        Ticket_Count=('Ticket_ID', 'count'),
                        Breach_Count=('Breach', 'sum'),
                        Avg_Duration=('Duration_Days', 'mean'),
                        Avg_Quality_Percent=('Quality_Percent', 'mean')
                    ).reset_index()
                    color_map_at = color_map_for(at_agg['Assigned_To'])

                    # Quality % bars
                    fig_a1 = px.bar(
                        at_agg.sort_values('Avg_Quality_Percent', ascending=False),
                        x='Assigned_To', y='Avg_Quality_Percent',
                        title='Quality % by Assigned-To',
                        color='Assigned_To', color_discrete_map=color_map_at,
                        text='Avg_Quality_Percent'
                    )
                    fig_a1.update_traces(texttemplate='%{text:.1f}%', hovertemplate='<b>%{x}</b><br>Quality: %{y:.1f}%<extra></extra>')
                    fig_a1.update_xaxes(title=None)
                    fig_a1.update_yaxes(title='Quality %', range=[0, 100])
                    fig_a1.add_hline(y=100, line_width=2, line_dash="dot", line_color="#888",
                                     annotation_text="Target 100%", annotation_position="top left")
                    st.plotly_chart(fig_a1, use_container_width=True)

                    # SLA Outcomes 100% stacked (manual)
                    fig_a2 = build_sla_100_stack(
                        at_agg.rename(columns={'Assigned_To': 'Entity'}),
                        entity_col='Entity',
                        title='SLA Outcomes by Assigned-To (100% Stacked)'
                    )
                    st.plotly_chart(fig_a2, use_container_width=True)

                    # Ticket Volume
                    fig_a3 = px.bar(
                        at_agg.sort_values('Ticket_Count', ascending=False),
                        x='Assigned_To', y='Ticket_Count',
                        title='Ticket Volume by Assigned-To',
                        color='Assigned_To', color_discrete_map=color_map_at,
                        text='Ticket_Count'
                    )
                    fig_a3.update_traces(texttemplate='%{text:.0f}', hovertemplate='<b>%{x}</b><br>Tickets: %{y:.0f}<extra></extra>')
                    fig_a3.update_xaxes(title=None)
                    fig_a3.update_yaxes(title='Ticket Count', tickformat=',d')
                    st.plotly_chart(fig_a3, use_container_width=True)

                # ========= Assignment Group =========
                st.subheader("👥 Assignment Group Performance")
                if 'Assignment_Group' in results_df.columns:
                    ag_agg = results_df.groupby('Assignment_Group').agg(
                        Ticket_Count=('Ticket_ID', 'count'),
                        Breach_Count=('Breach', 'sum'),
                        Avg_Duration=('Duration_Days', 'mean'),
                        Avg_Quality_Percent=('Quality_Percent', 'mean')
                    ).reset_index()
                    color_map_ag = color_map_for(ag_agg['Assignment_Group'])

                    fig_g1 = px.bar(
                        ag_agg.sort_values('Avg_Quality_Percent', ascending=False),
                        x='Assignment_Group', y='Avg_Quality_Percent',
                        title='Quality % by Assignment Group',
                        color='Assignment_Group', color_discrete_map=color_map_ag,
                        text='Avg_Quality_Percent'
                    )
                    fig_g1.update_traces(texttemplate='%{text:.1f}%', hovertemplate='<b>%{x}</b><br>Quality: %{y:.1f}%<extra></extra>')
                    fig_g1.update_xaxes(title=None)
                    fig_g1.update_yaxes(title='Quality %', range=[0, 100])
                    fig_g1.add_hline(y=100, line_width=2, line_dash="dot", line_color="#888",
                                     annotation_text="Target 100%", annotation_position="top left")
                    st.plotly_chart(fig_g1, use_container_width=True)

                    fig_g2 = build_sla_100_stack(
                        ag_agg.rename(columns={'Assignment_Group': 'Entity'}),
                        entity_col='Entity',
                        title='SLA Outcomes by Assignment Group (100% Stacked)'
                    )
                    st.plotly_chart(fig_g2, use_container_width=True)

                    fig_g3 = px.bar(
                        ag_agg.sort_values('Ticket_Count', ascending=False),
                        x='Assignment_Group', y='Ticket_Count',
                        title='Ticket Volume by Assignment Group',
                        color='Assignment_Group', color_discrete_map=color_map_ag,
                        text='Ticket_Count'
                    )
                    fig_g3.update_traces(texttemplate='%{text:.0f}', hovertemplate='<b>%{x}</b><br>Tickets: %{y:.0f}<extra></extra>')
                    fig_g3.update_xaxes(title=None)
                    fig_g3.update_yaxes(title='Ticket Count', tickformat=',d')
                    st.plotly_chart(fig_g3, use_container_width=True)

                # ========= Domain =========
                st.subheader("🌐 Domain Performance")
                if 'Domain' in results_df.columns:
                    d_agg = results_df.groupby('Domain').agg(
                        Ticket_Count=('Ticket_ID', 'count'),
                        Breach_Count=('Breach', 'sum'),
                        Avg_Duration=('Duration_Days', 'mean'),
                        Avg_Quality_Percent=('Quality_Percent', 'mean')
                    ).reset_index()
                    color_map_dom = color_map_for(d_agg['Domain'])

                    fig_d1 = px.bar(
                        d_agg.sort_values('Avg_Quality_Percent', ascending=False),
                        x='Domain', y='Avg_Quality_Percent',
                        title='Quality % by Domain',
                        color='Domain', color_discrete_map=color_map_dom,
                        text='Avg_Quality_Percent'
                    )
                    fig_d1.update_traces(texttemplate='%{text:.1f}%', hovertemplate='<b>%{x}</b><br>Quality: %{y:.1f}%<extra></extra>')
                    fig_d1.update_xaxes(title=None)
                    fig_d1.update_yaxes(title='Quality %', range=[0, 100])
                    fig_d1.add_hline(y=100, line_width=2, line_dash="dot", line_color="#888",
                                     annotation_text="Target 100%", annotation_position="top left")
                    st.plotly_chart(fig_d1, use_container_width=True)

                    fig_d2 = build_sla_100_stack(
                        d_agg.rename(columns={'Domain': 'Entity'}),
                        entity_col='Entity',
                        title='SLA Outcomes by Domain (100% Stacked)'
                    )
                    st.plotly_chart(fig_d2, use_container_width=True)

                    fig_d3 = px.bar(
                        d_agg.sort_values('Ticket_Count', ascending=False),
                        x='Domain', y='Ticket_Count',
                        title='Ticket Volume by Domain',
                        color='Domain', color_discrete_map=color_map_dom,
                        text='Ticket_Count'
                    )
                    fig_d3.update_traces(texttemplate='%{text:.0f}', hovertemplate='<b>%{x}</b><br>Tickets: %{y:.0f}<extra></extra>')
                    fig_d3.update_xaxes(title=None)
                    fig_d3.update_yaxes(title='Ticket Count', tickformat=',d')
                    st.plotly_chart(fig_d3, use_container_width=True)

                # ========= Distribution =========
                st.markdown("---")
                st.header("📦 Quality Distribution (by Quality %)")

                def band_quality(pct):
                    if pct >= 90: return 'Excellent (≥90%)'
                    if pct >= 75: return 'Good (75–89%)'
                    if pct >= 60: return 'Fair (60–74%)'
                    return 'Poor (<60%)'

                dist_df = results_df.copy()
                dist_df['Quality_Band'] = dist_df['Quality_Percent'].apply(band_quality)
                band_order = ['Excellent (≥90%)', 'Good (75–89%)', 'Fair (60–74%)', 'Poor (<60%)']
                dist_df['Quality_Band'] = pd.Categorical(dist_df['Quality_Band'], categories=band_order, ordered=True)

                band_counts = dist_df['Quality_Band'].value_counts().reindex(band_order).fillna(0).astype(int)
                band_pct = (band_counts / band_counts.sum() * 100).fillna(0)

                band_colors = {
                    'Excellent (≥90%)': '#2ca02c',
                    'Good (75–89%)': '#66bb6a',
                    'Fair (60–74%)': '#ff7f0e',
                    'Poor (<60%)': '#d62728'
                }

                chart_df = band_counts.reset_index()
                chart_df.columns = ['Quality_Band', 'Count']

                fig_dist = px.bar(
                    chart_df, x='Quality_Band', y='Count',
                    title='Quality Score Distribution (Banded)',
                    color='Quality_Band', color_discrete_map=band_colors,
                    text='Count'
                )
                fig_dist.update_traces(texttemplate='%{text:.0f}', hovertemplate='<b>%{x}</b><br>Tickets: %{y:.0f}<extra></extra>')
                fig_dist.update_xaxes(title=None)
                fig_dist.update_yaxes(title='Ticket Count', tickformat=',d')
                st.plotly_chart(fig_dist, use_container_width=True)

                st.markdown("**Summary:**")
                cols = st.columns(4)
                for i, band in enumerate(band_order):
                    with cols[i]:
                        st.metric(band, f"{band_counts.get(band, 0):,} ({band_pct.get(band, 0):.1f}%)")

                # ========= Detailed + Download =========
                st.markdown("---")
                st.header("📋 Detailed Audit Results")

                display_df = results_df.copy()
                display_df['Issues'] = display_df['Issues'].apply(lambda x: '; '.join(map(str, x)) if x else 'No issues')
                st.dataframe(display_df, use_container_width=True)

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    display_df.to_excel(writer, index=False, sheet_name='Audit Results')
                output.seek(0)
                st.download_button(
                    label="📥 Download Audit Report",
                    data=output,
                    file_name=f"ticket_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"❌ Error processing tickets: {str(e)}")
                st.exception(e)

else:
    st.info("👆 Upload your ticket file (and optional mapping) and click **Run Audit** to start.")
    st.markdown("---")
    st.header("📖 Accepted Columns (case-insensitive)")
    st.markdown("""
- **Number** → Ticket ID string  
- **Opened** → Creation date/time  
- **Resolved** → Resolution/close date/time (system)  
- **Closed** → Final closed date/time (used to derive **Closed Month** as `MMM YYYY`)  
- **Category**  
- **Work notes**, **Additional comments**  
- **Resolution notes** / **Resolution summary** / **Closure notes** (any one)  
- **Assignment group**, **Service offering**  
- **Reopen count**, **Reassignment count**  
- **Assigned to**, **Portfolio**, **Domain**

**Mapping file (optional, 1:N supported):**  
Column 1 → Assignment Group  
Column 2 → Service Offering  
If multiple rows exist for the same Assignment Group with different Service Offerings, **all** of them are accepted as valid.  
Correct AG↔SO adds **+1** to quality; if mapping not uploaded, max score is **5**.
""")

st.markdown("---")
st.markdown("**Ticket Quality Audit System — Rule-based**")
