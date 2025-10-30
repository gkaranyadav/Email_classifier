# app.py - Complete Email Classifier App
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="AI Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #10B981;
    }
    .email-card {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class DatabricksClient:
    def __init__(self):
        self.databricks_host = st.secrets.get("DATABRICKS_HOST")
        self.databricks_token = st.secrets.get("DATABRICKS_TOKEN")
        self.deepseek_api_key = st.secrets.get("DEEPSEEK_API_KEY")
        self.job_id = st.secrets.get("DATABRICKS_JOB_ID")
    
    def test_connection(self):
        """Test Databricks connection"""
        try:
            headers = {"Authorization": f"Bearer {self.databricks_token}"}
            response = requests.get(
                f"{self.databricks_host}/api/2.0/clusters/list",
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def run_email_job(self, gmail_token: str, max_emails: int = 5, email_data: dict = None):
        """Run the email classification job"""
        try:
            # Prepare job parameters
            notebook_params = {
                "gmail_token": gmail_token,
                "deepseek_api_key": self.deepseek_api_key,
                "max_emails": str(max_emails)
            }
            
            if email_data:
                notebook_params["email_data"] = json.dumps(email_data)
            
            # Submit job run
            job_payload = {
                "job_id": self.job_id,
                "notebook_params": notebook_params
            }
            
            headers = {
                "Authorization": f"Bearer {self.databricks_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.databricks_host}/api/2.1/jobs/run-now",
                headers=headers,
                json=job_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                run_id = response.json()["run_id"]
                return {"success": True, "run_id": run_id, "message": "Job started successfully!"}
            else:
                return {"success": False, "error": f"Job submission failed: {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": f"Error: {str(e)}"}
    
    def get_job_result(self, run_id: str):
        """Get job execution result"""
        try:
            headers = {
                "Authorization": f"Bearer {self.databricks_token}",
                "Content-Type": "application/json"
            }
            
            # Wait for job completion with progress
            for i in range(30):  # 30 attempts = 5 minutes max
                time.sleep(10)
                
                status_response = requests.get(
                    f"{self.databricks_host}/api/2.1/jobs/runs/get?run_id={run_id}",
                    headers=headers,
                    timeout=30
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    state = status_data["state"]
                    
                    if state["life_cycle_state"] == "TERMINATED":
                        if state["result_state"] == "SUCCESS":
                            # Get output
                            output_response = requests.get(
                                f"{self.databricks_host}/api/2.1/jobs/runs/get-output?run_id={run_id}",
                                headers=headers,
                                timeout=30
                            )
                            
                            if output_response.status_code == 200:
                                output_data = output_response.json()
                                if "notebook_output" in output_data:
                                    result = output_data["notebook_output"]["result"]
                                    return json.loads(result)
                                else:
                                    return {"success": False, "error": "No output from job"}
                        else:
                            return {"success": False, "error": f"Job failed: {state.get('state_message', 'Unknown error')}"}
                    elif state["life_cycle_state"] in ["INTERNAL_ERROR", "SKIPPED"]:
                        return {"success": False, "error": f"Job error: {state.get('state_message', 'Unknown error')}"}
            
            return {"success": False, "error": "Job timeout"}
            
        except Exception as e:
            return {"success": False, "error": f"Error getting result: {str(e)}"}

class EmailClassifierApp:
    def __init__(self):
        self.databricks_client = DatabricksClient()
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state"""
        if 'classifications' not in st.session_state:
            st.session_state.classifications = []
        if 'gmail_token' not in st.session_state:
            st.session_state.gmail_token = ""
        if 'processing' not in st.session_state:
            st.session_state.processing = False
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.title("🔐 Configuration")
            
            # Gmail Token Input
            st.subheader("Gmail Connection")
            gmail_token = st.text_input(
                "Gmail Access Token",
                value=st.session_state.gmail_token,
                type="password",
                placeholder="Paste your Gmail access token here...",
                help="Get from Google OAuth Playground"
            )
            
            if gmail_token != st.session_state.gmail_token:
                st.session_state.gmail_token = gmail_token
            
            # Test Connections
            st.subheader("🔧 Connection Status")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Test Databricks"):
                    if self.databricks_client.test_connection():
                        st.success("✅ Connected")
                    else:
                        st.error("❌ Failed")
            
            with col2:
                if st.button("Test Gmail"):
                    if st.session_state.gmail_token:
                        # Simple Gmail test
                        try:
                            headers = {
                                "Authorization": f"Bearer {st.session_state.gmail_token}",
                                "Content-Type": "application/json"
                            }
                            response = requests.get(
                                "https://gmail.googleapis.com/gmail/v1/users/me/labels",
                                headers=headers,
                                timeout=10
                            )
                            if response.status_code == 200:
                                st.success("✅ Gmail OK")
                            else:
                                st.error("❌ Gmail Failed")
                        except:
                            st.error("❌ Gmail Failed")
                    else:
                        st.warning("Enter token first")
            
            # Processing Controls
            st.subheader("📊 Processing")
            email_count = st.slider("Emails to process", 1, 10, 3)
            
            if st.button("🚀 Process Emails", type="primary", use_container_width=True):
                if st.session_state.gmail_token:
                    st.session_state.processing = True
                    self.process_emails(email_count)
                else:
                    st.error("Please enter Gmail token first")
            
            # Statistics
            st.subheader("📈 Statistics")
            st.write(f"Classifications: {len(st.session_state.classifications)}")
            if st.session_state.classifications:
                df = pd.DataFrame(st.session_state.classifications)
                high_priority = len(df[df['priority'] == 'High'])
                st.write(f"High Priority: {high_priority}")
    
    def process_emails(self, email_count):
        """Process emails through Databricks job"""
        try:
            with st.spinner("🔄 Starting email classification job..."):
                # Submit job to Databricks
                job_result = self.databricks_client.run_email_job(
                    st.session_state.gmail_token,
                    email_count
                )
                
                if job_result["success"]:
                    st.success(f"✅ {job_result['message']}")
                    st.info(f"Job Run ID: {job_result['run_id']}")
                    
                    # Wait for job completion
                    with st.spinner("⏳ Processing emails with AI..."):
                        final_result = self.databricks_client.get_job_result(job_result['run_id'])
                        
                        if final_result and final_result.get("success"):
                            processed_emails = final_result.get("emails_processed", [])
                            if processed_emails:
                                st.session_state.classifications.extend(processed_emails)
                                st.success(f"✅ Processed {len(processed_emails)} emails!")
                            else:
                                st.info("📭 No emails found or processed")
                        else:
                            st.error(f"❌ Job failed: {final_result.get('error', 'Unknown error')}")
                else:
                    st.error(f"❌ {job_result.get('error', 'Job failed')}")
        
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
        finally:
            st.session_state.processing = False
    
    def render_email_tab(self):
        """Render email classification tab"""
        st.header("📧 Email Classification")
        
        if not st.session_state.gmail_token:
            st.info("👆 Enter your Gmail token in the sidebar to start")
            return
        
        # Manual classification
        st.subheader("Manual Classification")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            subject = st.text_input("Email Subject", placeholder="Order Issue #12345")
            sender = st.text_input("Sender", placeholder="customer@example.com")
        
        with col2:
            email_body = st.text_area("Email Content", height=150, 
                                    placeholder="Hello, I'm having issues with my order...",
                                    value="Hello, I'm having issues with my recent order. The product arrived damaged and doesn't work properly. I would like to request a refund or replacement.")
        
        if st.button("🤖 Classify This Email", type="secondary"):
            if subject and email_body:
                with st.spinner("Analyzing email..."):
                    email_data = {
                        'subject': subject,
                        'from': sender or 'customer@example.com',
                        'body': email_body
                    }
                    
                    job_result = self.databricks_client.run_email_job(
                        st.session_state.gmail_token,
                        max_emails=1,
                        email_data=email_data
                    )
                    
                    if job_result["success"]:
                        final_result = self.databricks_client.get_job_result(job_result['run_id'])
                        if final_result and final_result.get("success"):
                            processed = final_result.get("emails_processed", [])
                            if processed:
                                st.session_state.classifications.extend(processed)
                                self.display_classification_result(processed[0])
                            else:
                                st.error("No result from classification")
                        else:
                            st.error(f"Classification failed: {final_result.get('error', 'Unknown error')}")
                    else:
                        st.error(f"Job submission failed: {job_result.get('error', 'Unknown error')}")
            else:
                st.warning("Please enter subject and email content")
        
        # Display recent classifications
        if st.session_state.classifications:
            st.subheader("Recent Classifications")
            for result in reversed(st.session_state.classifications[-5:]):
                self.display_email_card(result)
    
    def display_classification_result(self, result):
        """Display classification result"""
        with st.container():
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Category", result['category'])
            with col2:
                priority_color = {"High": "red", "Medium": "orange", "Low": "green"}
                st.metric("Priority", result['priority'])
            with col3:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            
            # AI Response
            with st.expander("📋 AI Response", expanded=True):
                st.write(result['reply'])
            
            st.markdown("---")
    
    def display_email_card(self, result):
        """Display email classification card"""
        priority_colors = {
            "High": "#FFEBEE",
            "Medium": "#FFF3E0", 
            "Low": "#E8F5E8"
        }
        
        st.markdown(f"""
        <div class="email-card">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 1;">
                    <strong>{result['subject']}</strong>
                    <div style="font-size: 0.9em; color: #666; margin-top: 0.25rem;">
                        From: {result.get('from', 'Unknown')}
                    </div>
                </div>
                <span style="background: {priority_colors.get(result['priority'], '#F5F5F5')}; 
                            padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8em; 
                            font-weight: bold; border: 1px solid #E5E7EB;">
                    {result['priority']}
                </span>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.9em;">
                <strong>Category:</strong> {result['category']} • 
                <strong>Sentiment:</strong> {result.get('sentiment', 'N/A')} •
                <strong>Confidence:</strong> {result['confidence']:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_analytics_tab(self):
        """Render analytics dashboard"""
        st.header("📊 Analytics Dashboard")
        
        if not st.session_state.classifications:
            st.info("No data available. Classify some emails first.")
            return
        
        df = pd.DataFrame(st.session_state.classifications)
        
        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Emails", len(df))
        with col2:
            high_priority = len(df[df['priority'] == 'High'])
            st.metric("High Priority", high_priority)
        with col3:
            categories = df['category'].nunique()
            st.metric("Categories", categories)
        with col4:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Email Categories")
            if 'category' in df.columns:
                category_counts = df['category'].value_counts()
                fig = px.pie(values=category_counts.values, names=category_counts.index)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Priority Distribution")
            if 'priority' in df.columns:
                priority_counts = df['priority'].value_counts()
                fig = px.bar(x=priority_counts.index, y=priority_counts.values,
                            labels={'x': 'Priority', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🤖 AI Email Classifier</h1>', unsafe_allow_html=True)
        
        # Check if Databricks client is initialized
        if not all([self.databricks_client.databricks_host, self.databricks_client.databricks_token]):
            st.error("❌ Databricks credentials not configured. Check your secrets.toml file.")
            return
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2 = st.tabs(["📧 Classify Emails", "📊 Analytics"])
        
        with tab1:
            self.render_email_tab()
        
        with tab2:
            self.render_analytics_tab()

# Run the application
if __name__ == "__main__":
    app = EmailClassifierApp()
    app.run()
