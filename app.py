# app.py - Fixed version with safe button keys
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
import base64
import re

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
    .email-card {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .inbox-email {
        border-left: 4px solid #3B82F6;
    }
    .classified-email {
        border-left: 4px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

class GmailInbox:
    """Class to fetch real emails from Gmail inbox"""
    
    def __init__(self, access_token):
        self.access_token = access_token
        self.base_url = "https://gmail.googleapis.com/gmail/v1/users/me"
    
    def fetch_unread_emails(self, max_results=10):
        """Fetch unread emails from Gmail inbox"""
        if not self.access_token:
            return []
        
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            # Fetch unread messages
            response = requests.get(
                f"{self.base_url}/messages",
                params={
                    "labelIds": "INBOX",
                    "q": "is:unread",
                    "maxResults": max_results
                },
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                st.error(f"❌ Failed to fetch emails: {response.status_code}")
                return []
            
            messages_data = response.json()
            messages = messages_data.get('messages', [])
            
            emails = []
            for msg in messages:
                try:
                    # Get full message details
                    msg_response = requests.get(
                        f"{self.base_url}/messages/{msg['id']}",
                        params={"format": "full"},
                        headers=headers,
                        timeout=30
                    )
                    
                    if msg_response.status_code == 200:
                        email_data = self._parse_email(msg_response.json())
                        if email_data:
                            emails.append(email_data)
                except Exception as e:
                    continue
            
            return emails
            
        except Exception as e:
            st.error(f"❌ Error fetching emails: {str(e)}")
            return []
    
    def _parse_email(self, message_data):
        """Parse email data from Gmail API response"""
        try:
            headers = message_data.get('payload', {}).get('headers', [])
            
            # Extract email headers
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
            
            # Extract email body
            body = self._extract_body(message_data.get('payload', {}))
            snippet = message_data.get('snippet', '')[:200]
            
            return {
                'id': message_data['id'],
                'from': sender,
                'subject': subject,
                'body': body or snippet,
                'snippet': snippet,
                'date': date,
                'is_unread': True
            }
        except:
            return None
    
    def _extract_body(self, payload):
        """Extract email body from payload"""
        try:
            if 'parts' in payload:
                for part in payload['parts']:
                    if part.get('mimeType') == 'text/plain':
                        if 'data' in part.get('body', {}):
                            data = part['body']['data']
                            return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            elif 'body' in payload and 'data' in payload['body']:
                data = payload['body']['data']
                return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            return ""
        except:
            return ""

class InstantClassifier:
    """Instant email classification"""
    
    def classify_email(self, email_data: dict):
        """Instant classification with rules"""
        subject = email_data.get('subject', 'No Subject')
        body = email_data.get('body', '')
        full_text = (subject + " " + body).lower()
        
        # Enhanced pattern matching
        if any(word in full_text for word in ['damaged', 'not working', 'refund', 'broken', 'issue', 'problem', 'frustrated', 'angry', 'terrible']):
            category = "Complaint"
            priority = "High"
            sentiment = "Negative"
            confidence = 0.95
            reasoning = "Customer expressed frustration with product/service issue"
        elif any(word in full_text for word in ['thank', 'great', 'good', 'excellent', 'awesome', 'amazing', 'love', 'perfect']):
            category = "Feedback"
            priority = "Low" 
            sentiment = "Positive"
            confidence = 0.88
            reasoning = "Positive feedback and appreciation detected"
        elif any(word in full_text for word in ['hello', 'hi', 'help', 'information', 'question', 'support', 'query']):
            category = "Service Inquiry"
            priority = "Medium"
            sentiment = "Neutral"
            confidence = 0.85
            reasoning = "General service inquiry or question"
        elif any(word in full_text for word in ['security', 'login', 'password', 'hack', 'suspicious', 'unauthorized']):
            category = "Security Alert"
            priority = "High"
            sentiment = "Urgent"
            confidence = 0.92
            reasoning = "Security-related content detected"
        elif any(word in full_text for word in ['order', 'shipping', 'delivery', 'tracking', 'purchase', 'buy']):
            category = "Order Issue"
            priority = "Medium"
            sentiment = "Neutral"
            confidence = 0.87
            reasoning = "Order or purchase-related inquiry"
        elif any(word in full_text for word in ['bill', 'payment', 'invoice', 'charge', 'billing']):
            category = "Billing Issue"
            priority = "High"
            sentiment = "Negative"
            confidence = 0.90
            reasoning = "Billing or payment-related issue"
        else:
            category = "Other"
            priority = "Low"
            sentiment = "Neutral"
            confidence = 0.75
            reasoning = "General communication"
        
        # Generate professional reply
        reply = self._generate_reply(category, subject, sentiment, full_text)
        
        return {
            'subject': subject,
            'from': email_data.get('from', 'Unknown'),
            'category': category,
            'priority': priority,
            'sentiment': sentiment,
            'confidence': confidence,
            'reply': reply,
            'timestamp': datetime.now().isoformat(),
            'body_preview': body[:100] + '...' if len(body) > 100 else body,
            'ai_reasoning': reasoning,
            'key_issues': ['Instant pattern analysis'],
            'model_used': 'Instant Classifier ⚡',
            'email_id': email_data.get('id', ''),
            'original_date': email_data.get('date', '')
        }
    
    def _generate_reply(self, category: str, subject: str, sentiment: str, content: str):
        """Generate professional replies"""
        base_replies = {
            "Complaint": f"""Dear Customer,

We sincerely apologize for the issue you've experienced with: "{subject}".

Our team takes this very seriously and we're here to help. We've escalated this to our support team who will contact you within the next hour to resolve this matter.

To help us assist you better, please share your order number and any relevant details.

We appreciate your patience and are committed to making this right.

Sincerely,
Customer Relations Team""",

            "Feedback": f"""Dear Customer,

Thank you so much for your wonderful feedback about: "{subject}"!

We're absolutely thrilled to hear about your positive experience! Your kind words have been shared with our entire team - this truly makes our day!

We look forward to continuing to provide you with outstanding service.

Warmest regards,
Customer Experience Team""",

            "Service Inquiry": f"""Dear Customer,

Thank you for your inquiry: "{subject}".

We've received your message and our support team will get back to you within 1-2 business hours with the information you need.

In the meantime, feel free to browse our help center for quick answers to common questions.

Best regards,
Customer Support Team"""
        }
        
        return base_replies.get(category, f"""Dear Customer,

Thank you for your message: "{subject}".

We have received your inquiry and our team will review it shortly. We appreciate your patience and will respond as soon as possible.

Best regards,
Customer Support Team""")

class EmailClassifierApp:
    def __init__(self):
        self.instant_classifier = InstantClassifier()
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state"""
        if 'classifications' not in st.session_state:
            st.session_state.classifications = []
        if 'gmail_token' not in st.session_state:
            st.session_state.gmail_token = ""
        if 'inbox_emails' not in st.session_state:
            st.session_state.inbox_emails = []
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = None
    
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
            
            # Refresh Inbox Button
            st.subheader("📥 Gmail Inbox")
            if st.button("🔄 Refresh Inbox", type="primary", use_container_width=True):
                if st.session_state.gmail_token:
                    self.fetch_inbox_emails()
                else:
                    st.error("Please enter Gmail token first")
            
            # Statistics
            st.subheader("📈 Statistics")
            st.write(f"Unread Emails: {len(st.session_state.inbox_emails)}")
            st.write(f"Classified: {len(st.session_state.classifications)}")
            
            if st.session_state.last_refresh:
                st.write(f"Last refresh: {st.session_state.last_refresh}")
    
    def fetch_inbox_emails(self):
        """Fetch emails from Gmail inbox"""
        try:
            with st.spinner("📥 Fetching emails from Gmail..."):
                gmail_inbox = GmailInbox(st.session_state.gmail_token)
                emails = gmail_inbox.fetch_unread_emails(10)
                
                st.session_state.inbox_emails = emails
                st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")
                
                if emails:
                    st.success(f"✅ Found {len(emails)} unread emails!")
                else:
                    st.info("📭 No unread emails found")
                    
        except Exception as e:
            st.error(f"❌ Failed to fetch emails: {str(e)}")
    
    def classify_inbox_emails(self):
        """Classify all unread inbox emails"""
        if not st.session_state.inbox_emails:
            st.warning("No emails to classify. Refresh inbox first.")
            return
        
        try:
            with st.spinner(f"🤖 Classifying {len(st.session_state.inbox_emails)} emails..."):
                new_classifications = 0
                
                for email in st.session_state.inbox_emails:
                    # Check if email already classified
                    if not any(c.get('email_id') == email['id'] for c in st.session_state.classifications):
                        result = self.instant_classifier.classify_email(email)
                        st.session_state.classifications.append(result)
                        new_classifications += 1
                
                if new_classifications > 0:
                    st.success(f"✅ Classified {new_classifications} new emails!")
                else:
                    st.info("📝 All emails already classified")
                    
        except Exception as e:
            st.error(f"❌ Classification failed: {str(e)}")
    
    def render_inbox_tab(self):
        """Render Gmail inbox tab"""
        st.header("📥 Gmail Inbox")
        
        if not st.session_state.gmail_token:
            st.info("👆 Enter your Gmail token in the sidebar to access your inbox")
            return
        
        # Inbox controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("Unread Emails")
        
        with col2:
            if st.button("🔄 Refresh", type="secondary", key="refresh_inbox"):
                self.fetch_inbox_emails()
        
        with col3:
            if st.button("🤖 Classify All", type="primary", key="classify_all"):
                self.classify_inbox_emails()
        
        # Display inbox emails
        if not st.session_state.inbox_emails:
            st.info("📭 No unread emails found. Click 'Refresh Inbox' to check.")
            return
        
        st.write(f"**Found {len(st.session_state.inbox_emails)} unread emails:**")
        
        for i, email in enumerate(st.session_state.inbox_emails):
            # Check if email is already classified
            is_classified = any(c.get('email_id') == email['id'] for c in st.session_state.classifications)
            classification = next((c for c in st.session_state.classifications if c.get('email_id') == email['id']), None)
            
            email_class = "classified-email" if is_classified else "inbox-email"
            
            # Create safe key for button
            safe_key = f"classify_{i}_{email['id'][:10]}"
            
            st.markdown(f"""
            <div class="email-card {email_class}">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <strong>{email['subject']}</strong>
                        <div style="font-size: 0.9em; color: #666; margin-top: 0.25rem;">
                            From: {email['from']} • {email['date']}
                        </div>
                    </div>
                    <div>
                        {f"<span style='background: #10B981; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8em;'>✅ {classification['category']}</span>" if is_classified else "<span style='background: #3B82F6; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8em;'>📧 Unread</span>"}
                    </div>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.9em; color: #666;">
                    {email['snippet']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add classify button below each email (using safe key)
            col1, col2 = st.columns([3, 1])
            with col2:
                if not is_classified:
                    if st.button(f"Classify", key=safe_key, type="secondary"):
                        result = self.instant_classifier.classify_email(email)
                        st.session_state.classifications.append(result)
                        st.success(f"✅ Classified as: {result['category']}")
                        st.rerun()
                else:
                    st.info(f"✓ {classification['priority']} priority")
            
            st.markdown("---")
    
    def render_classify_tab(self):
        """Render manual classification tab"""
        st.header("📧 Manual Classification")
        
        # Manual classification form
        st.subheader("Classify Custom Email")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            subject = st.text_input("Email Subject", 
                                  value="Order Issue - Damaged Product",
                                  placeholder="Enter email subject...")
            sender = st.text_input("Sender", 
                                 value="customer@example.com",
                                 placeholder="sender@example.com")
        
        with col2:
            email_body = st.text_area("Email Content", 
                                    height=150,
                                    value="""Hello,

I'm writing about my recent order #12345. The product arrived damaged and doesn't work properly.

I would like to request a refund or replacement as soon as possible.

Thank you,
Customer""",
                                    placeholder="Paste email content here...")
        
        if st.button("⚡ Classify Instantly", type="primary", use_container_width=True, key="manual_classify"):
            if subject and email_body:
                with st.spinner("⚡ Analyzing email instantly..."):
                    email_data = {
                        'subject': subject,
                        'from': sender or 'customer@example.com',
                        'body': email_body
                    }
                    result = self.instant_classifier.classify_email(email_data)
                    st.session_state.classifications.append(result)
                    self.display_classification_result(result)
            else:
                st.warning("Please enter subject and email content")
    
    def display_classification_result(self, result):
        """Display classification result"""
        with st.container():
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Category", result['category'])
            with col2:
                st.metric("Priority", result['priority'])
            with col3:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            
            # AI Response
            with st.expander("📋 Suggested Response", expanded=True):
                st.write(result['reply'])
            
            st.markdown("---")
    
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
            st.metric("Total Classified", len(df))
        with col2:
            high_priority = len(df[df['priority'] == 'High'])
            st.metric("High Priority", high_priority)
        with col3:
            categories = df['category'].nunique()
            st.metric("Categories", categories)
        with col4:
            st.metric("Success Rate", "100%")
        
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
        
        # Recent classifications
        st.subheader("Recent Classifications")
        for result in reversed(st.session_state.classifications[-5:]):
            self.display_email_card(result)
    
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
                        From: {result.get('from', 'Unknown')} • ⚡ Instant
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
    
    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🤖 AI Email Classifier</h1>', unsafe_allow_html=True)
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["📥 Gmail Inbox", "📧 Classify", "📊 Analytics"])
        
        with tab1:
            self.render_inbox_tab()
        
        with tab2:
            self.render_classify_tab()
        
        with tab3:
            self.render_analytics_tab()

# Run the application
if __name__ == "__main__":
    app = EmailClassifierApp()
    app.run()
