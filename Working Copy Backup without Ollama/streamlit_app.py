"""
Pharmacy AI - Multi-Agent Platform
GUARANTEED WORKING Auto-Fill - Uses Forms
"""

import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime
import io

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from master_agent import MasterAgent

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pharmacy AI Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .description-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">🏥 Pharmacy AI - Multi-Agent Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intelligent Decision Support for Pharmacy Operations</div>', unsafe_allow_html=True)

# ============================================================
# PROJECT DESCRIPTION
# ============================================================

st.markdown("""
<div class="description-box">
    <h3 style="margin-top: 0; color: #1f77b4;">📋 About This System</h3>
    <p style="margin-bottom: 0.5rem;">
        This AI-powered platform helps pharmacy operations make data-driven decisions across <b>10 critical areas</b>:
    </p>
    <ul style="margin-bottom: 0;">
        <li><b>Demand Forecasting:</b> Predict medicine demand with weather, seasonal & doctor patterns</li>
        <li><b>Store Transfers:</b> Optimize inventory distribution to prevent expiry & stockouts</li>
        <li><b>Supplier Selection:</b> Evaluate suppliers on reliability, cost, quality & compliance</li>
        <li><b>Working Capital:</b> Validate budgets, calculate DIO, forecast cash flow</li>
        <li><b>Inventory Optimization:</b> Calculate safety stock, identify dead stock, reorder points</li>
        <li><b>Pricing & Discounts:</b> Analyze elasticity, competitor prices, optimize margins</li>
        <li><b>Prescription Intelligence:</b> Understand doctor prescribing patterns by location</li>
        <li><b>Promotion ROI:</b> Measure campaign effectiveness and returns</li>
        <li><b>Compliance:</b> Monitor storage, expiry, controlled drugs & regulations</li>
        <li><b>Customer Personalization:</b> Recommend products & refill reminders</li>
    </ul>
    <p style="margin-top: 1rem; margin-bottom: 0; color: #28a745; font-weight: bold;">
        ✓ Zero operational cost • ✓ Natural language questions • ✓ Multi-agent coordination
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INITIALIZE MASTER AGENT
# ============================================================

if 'master_agent' not in st.session_state:
    with st.spinner("🔄 Loading AI agents..."):
        try:
            st.session_state.master_agent = MasterAgent()
            st.session_state.chat_history = []
        except Exception as e:
            st.error(f"❌ Error loading system: {str(e)}")
            st.stop()

# Initialize the question in session state if not present
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

# ============================================================
# SIDEBAR - SAMPLE QUESTIONS
# ============================================================

with st.sidebar:
    st.markdown("### 💡 Sample Questions")
    st.markdown("Click any question to copy to prompt:")
    
    sample_questions = [
        "What will be demand for Paracetamol next month?",
        "Which supplier should I use for Ibuprofen?",
        "Should I order 1000 units of Amoxicillin?",
        "What items are about to expire in next 30 days?",
        "Which products should I discount this week?",
        "What is my current DIO?",
        "Show me dead stock items",
        "Which medicines are overstocked?",
        "Recommend inter-store transfers to prevent expiry",
        "What is the ROI of last month's promotion?",
    ]
    
    for i, q in enumerate(sample_questions):
        if st.button(f"📌 {q}", key=f"sq_{i}", use_container_width=True):
            st.session_state.current_question = q
            st.rerun()
    
    st.markdown("---")
    
    # System Status
    st.markdown("### 📊 System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Status:**")
        st.markdown("🟢 **Online**")
    with col2:
        st.markdown("**Cost:**")
        st.markdown("**$0/month**")
    
    # Agent Status
    with st.expander("🤖 View All Agents"):
        if hasattr(st.session_state.master_agent, 'available_agents'):
            agents = st.session_state.master_agent.available_agents
            agent_names = [
                ("demand", "Demand Forecasting"),
                ("transfer", "Store Transfer"),
                ("supplier", "Supplier Intelligence"),
                ("capital", "Working Capital"),
                ("inventory", "Inventory Optimization"),
                ("pricing", "Discount & Pricing"),
                ("prescription", "Prescription Intelligence"),
                ("promotion", "Promotion Effectiveness"),
                ("compliance", "Compliance & Regulation"),
                ("customer", "Customer Personalization"),
            ]
            
            for key, name in agent_names:
                status = "✅" if agents.get(key) else "❌"
                st.text(f"{status} {name}")

# ============================================================
# MAIN AREA - QUESTION INPUT WITH FORM
# ============================================================

st.markdown("### 🤔 Ask Your Question")

# Use a form for better control
with st.form(key='question_form', clear_on_submit=False):
    # The text input - no key parameter, just default_value
    question_input = st.text_area(
        "Type your question or click a sample question from the sidebar:",
        value=st.session_state.current_question,
        height=100,
        placeholder="e.g., What will be demand for Paracetamol?"
    )
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        ask_button = st.form_submit_button("🚀 Ask Question", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.form_submit_button("🗑️ Clear", use_container_width=True)

# Handle clear button
if clear_button:
    st.session_state.chat_history = []
    st.session_state.current_question = ""
    st.rerun()

# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button and question_input:
    with st.spinner("🤖 Analyzing question and consulting AI agents..."):
        try:
            # Call master agent
            result = st.session_state.master_agent.ask_concise(question_input)
            
            # Add to history
            st.session_state.chat_history.append({
                'question': question_input,
                'result': result,
                'timestamp': datetime.now()
            })
            
            # Clear the current question
            st.session_state.current_question = ""
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📊 Results")
    
    # Show latest result first
    for idx in range(len(st.session_state.chat_history) - 1, -1, -1):
        entry = st.session_state.chat_history[idx]
        
        with st.expander(
            f"Q{idx+1}: {entry['question'][:80]}..." if len(entry['question']) > 80 else f"Q{idx+1}: {entry['question']}",
            expanded=(idx == len(st.session_state.chat_history) - 1)
        ):
            st.markdown(f"**Question:** {entry['question']}")
            st.caption(f"⏰ {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            result = entry['result']
            
            # Display answer text
            if 'answer' in result:
                st.markdown("**Answer:**")
                st.info(result['answer'])
            
            # Display table if available
            if 'table_data' in result and result['table_data'] is not None:
                st.markdown("**Detailed Results:**")
                df = pd.DataFrame(result['table_data'])
                st.dataframe(df, use_container_width=True)
                
                # Download buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"result_{idx+1}.csv",
                        mime="text/csv",
                        key=f"csv_{idx}",
                        use_container_width=True
                    )
                
                with col2:
                    text_output = f"Question: {entry['question']}\n\n"
                    text_output += f"Answer: {result.get('answer', '')}\n\n"
                    text_output += "Detailed Results:\n"
                    text_output += df.to_string(index=False)
                    
                    st.download_button(
                        label="📥 Download TXT",
                        data=text_output,
                        file_name=f"result_{idx+1}.txt",
                        mime="text/plain",
                        key=f"txt_{idx}",
                        use_container_width=True
                    )
                
                with col3:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Results')
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"result_{idx+1}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"excel_{idx}",
                        use_container_width=True
                    )
            
            # Display metrics if available
            if 'metrics' in result and result['metrics']:
                st.markdown("**Key Metrics:**")
                metric_cols = st.columns(len(result['metrics']))
                for i, (label, value) in enumerate(result['metrics'].items()):
                    with metric_cols[i]:
                        st.metric(label, value)
            
            # Show which agents were consulted
            if 'agents_consulted' in result:
                st.caption(f"🤖 Agents: {', '.join(result['agents_consulted'])}")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("💰 **Cost:** $0/month")
with col2:
    st.markdown("🚀 **Status:** Production")
with col3:
    st.markdown("🤖 **Agents:** 10 Active")
with col4:
    st.markdown("✨ **Version:** 2.0")
