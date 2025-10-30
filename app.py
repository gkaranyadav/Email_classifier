# app.py - Streamlit Frontend for Email Classifier
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from databricks_client import DatabricksClient

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
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
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

class EmailClassifierApp:
    def __init__(self):
        self.databricks_client = DatabricksClient()
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'gmail_connected' not in st.session_state:
            st.session_state.gmail_connected = False
        if 'classifications' not in st.session_state:
            st.session_state.classifications = []
        if 'gmail_token' not in st.session_state:
            st.session_state.gmail_token = ""
        if 'processing' not in st.session_state:
            st.session_state.processing = False
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        with st.sidebar:
            st.title("🔐 Configuration")
            
            # Gmail API Token Input
            st.subheader("Gmail Connection")
            gmail_token = st.text_input(
                "Gmail Access Token",
                type="password",
                placeholder="Paste your Gmail API token here...",
                help="Get token from Google OAuth Playground"
            )
            
            if gmail_token and gmail_token != st.session_state.gmail_token:
                st.session_state.gmail_token = gmail_token
                if st.button("🔗 Connect Gmail", type="primary"):
                    with st.spinner("Connecting to Gmail..."):
                        success = self.databricks_client.connect_gmail(gmail_token)
                        if success:
                            st.session_state.gmail_connected = True
                            st.success("✅ Gmail connected!")
                            st.rerun()
                        else:
                            st.error("❌ Gmail connection failed")
            
            # Processing Controls
            st.subheader("📊 Processing")
            if st.session_state.gmail_connected:
                email_count = st.slider("Emails to process", 1, 20, 5)
                if st.button("🚀 Process Emails", type="primary"):
                    st.session_state.processing = True
                    self.process_emails(email_count)
            
            # Statistics
            st.subheader("📈 Statistics")
            st.write(f"Classifications: {len(st.session_state.classifications)}")
            if st.session_state.classifications:
                df = pd.DataFrame(st.session_state.classifications)
                high_priority = len(df[df['priority'] == 'High'])
                st.write(f"High Priority: {high_priority}")
    
    def process_emails(self, email_count):
        """Process emails through Databricks"""
        try:
            with st.spinner("🔄 Fetching and classifying emails..."):
                # Get unread emails via Databricks
                emails = self.databricks_client.get_unread_emails(email_count)
                
                if emails:
                    # Classify each email
                    for email in emails:
                        result = self.databricks_client.classify_email(email)
                        if result:
                            st.session_state.classifications.append(result)
                    
                    st.success(f"✅ Processed {len(emails)} emails!")
                else:
                    st.info("📭 No unread emails found")
        
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
        finally:
            st.session_state.processing = False
    
    def render_email_tab(self):
        """Render email classification tab"""
        st.header("📧 Email Classification")
        
        if not st.session_state.gmail_connected:
            st.info("👆 Connect Gmail in the sidebar to start classifying emails")
            return
        
        # Manual classification
        st.subheader("Manual Classification")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            subject = st.text_input("Email Subject", placeholder="Enter email subject...")
            sender = st.text_input("Sender", placeholder="sender@example.com")
        
        with col2:
            email_body = st.text_area("Email Content", height=150, 
                                    placeholder="Paste email content here...")
        
        if st.button("🤖 Classify This Email", type="secondary"):
            if subject and email_body:
                with st.spinner("Analyzing email..."):
                    email_data = {
                        'subject': subject,
                        'from': sender or 'manual@input.com',
                        'body': email_body
                    }
                    result = self.databricks_client.classify_email(email_data)
                    if result:
                        st.session_state.classifications.append(result)
                        self.display_classification_result(result)
                    else:
                        st.error("Classification failed")
            else:
                st.warning("Please enter subject and email content")
        
        # Display recent classifications
        if st.session_state.classifications:
            st.subheader("Recent Classifications")
            for result in reversed(st.session_state.classifications[-5:]):
                self.display_email_card(result)
    
    def display_classification_result(self, result):
        """Display classification result in a nice format"""
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
        """Display email card in classification history"""
        with st.container():
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
                    <strong>Sentiment:</strong> {result.get('sentiment', 'N/A')}
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
            category_counts = df['category'].value_counts()
            fig = px.pie(values=category_counts.values, names=category_counts.index)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Priority Distribution")
            priority_counts = df['priority'].value_counts()
            fig = px.bar(x=priority_counts.index, y=priority_counts.values,
                        labels={'x': 'Priority', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("Classification History")
        display_df = df[['subject', 'from', 'category', 'priority', 'confidence', 'timestamp']].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True)
    
    def render_settings_tab(self):
        """Render settings and documentation"""
        st.header("⚙️ Settings & Documentation")
        
        st.subheader("🔐 API Configuration")
        st.info("""
        **Required APIs:**
        - **Gmail API**: Get token from [Google OAuth Playground](https://developers.google.com/oauthplayground)
        - **DeepSeek API**: Get free API key from [DeepSeek Platform](https://platform.deepseek.com/)
        - **Databricks**: Configured in streamlit secrets
        """)
        
        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Connect Gmail**: Paste your Gmail API token in the sidebar
        2. **Process Emails**: Click 'Process Emails' to classify unread emails
        3. **Manual Classification**: Use the manual form for specific emails
        4. **View Analytics**: Check the analytics tab for insights
        """)
        
        # Test connections
        st.subheader("🔧 Connection Status")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Test Databricks Connection"):
                if self.databricks_client.test_connection():
                    st.success("✅ Databricks connected")
                else:
                    st.error("❌ Databricks connection failed")
        
        with col2:
            if st.button("Test DeepSeek API"):
                if self.databricks_client.test_deepseek():
                    st.success("✅ DeepSeek API working")
                else:
                    st.error("❌ DeepSeek API failed")
    
    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🤖 AI Email Classifier</h1>', unsafe_allow_html=True)
        
        # Initialize Databricks client
        if not self.databricks_client.initialize():
            st.error("❌ Failed to initialize Databricks client. Check your secrets.")
            return
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["📧 Classify Emails", "📊 Analytics", "⚙️ Settings"])
        
        with tab1:
            self.render_email_tab()
        
        with tab2:
            self.render_analytics_tab()
        
        with tab3:
            self.render_settings_tab()

# Run the application
if __name__ == "__main__":
    app = EmailClassifierApp()
    app.run()
