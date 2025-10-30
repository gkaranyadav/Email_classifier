# app.py - Complete with DeepSeek AI Integration
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
import base64

# Page configuration
st.set_page_config(
    page_title="AI Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DeepSeekAI:
    """DeepSeek AI integration for real email analysis"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/chat/completions"
    
    def analyze_email(self, subject, body):
        """Analyze email using DeepSeek AI"""
        if not self.api_key:
            return self._fallback_analysis(subject, body)
        
        try:
            prompt = f"""
            Analyze this email and provide a JSON response:

            EMAIL SUBJECT: {subject}
            EMAIL BODY: {body}

            Please analyze and respond with ONLY valid JSON in this exact format:
            {{
                "category": "Complaint/Feedback/Service Inquiry/Technical Support/Refund Request/Order Issue/Billing Issue/Security Alert/Account Issue/Product Inquiry/Other",
                "priority": "High/Medium/Low",
                "sentiment": "Positive/Negative/Neutral/Urgent",
                "confidence": 0.95,
                "reasoning": "Brief explanation of your analysis",
                "suggested_reply": "Professional response to send to the customer"
            }}

            Rules:
            - Complaint: Customer is unhappy, angry, frustrated, wants refund
            - Feedback: Positive comments, thanks, appreciation
            - Service Inquiry: General questions, help requests
            - Be accurate based on the actual email content
            """

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1000,
                "stream": False
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                # Clean JSON response
                content = content.replace('```json', '').replace('```', '').strip()
                
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # If JSON parsing fails, use fallback
                    return self._parse_ai_response(content, subject, body)
            else:
                st.error(f"DeepSeek API error: {response.status_code}")
                return self._fallback_analysis(subject, body)
                
        except Exception as e:
            st.error(f"AI analysis failed: {str(e)}")
            return self._fallback_analysis(subject, body)
    
    def _parse_ai_response(self, text, subject, body):
        """Parse AI text response when JSON fails"""
        # Extract category from AI response
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['complaint', 'angry', 'frustrated', 'worst', 'terrible', 'refund']):
            category = "Complaint"
            priority = "High"
            sentiment = "Negative"
        elif any(word in text_lower for word in ['feedback', 'positive', 'thank', 'good', 'great', 'excellent']):
            category = "Feedback"
            priority = "Low"
            sentiment = "Positive"
        elif any(word in text_lower for word in ['inquiry', 'question', 'help', 'information']):
            category = "Service Inquiry"
            priority = "Medium"
            sentiment = "Neutral"
        else:
            category = "Other"
            priority = "Low"
            sentiment = "Neutral"
        
        return {
            "category": category,
            "priority": priority,
            "sentiment": sentiment,
            "confidence": 0.85,
            "reasoning": f"AI Analysis: {text[:150]}...",
            "suggested_reply": self._generate_smart_reply(category, subject, body)
        }
    
    def _fallback_analysis(self, subject, body):
        """Fallback analysis when AI fails"""
        full_text = (subject + " " + body).lower()
        
        if any(word in full_text for word in ['worst', 'terrible', 'awful', '1/5', 'one star', 'refund', 'angry']):
            category = "Complaint"
            priority = "High"
            sentiment = "Negative"
            reasoning = "Customer expressed strong dissatisfaction"
        elif any(word in full_text for word in ['thank', 'great', 'good', 'excellent', 'awesome']):
            category = "Feedback"
            priority = "Low"
            sentiment = "Positive"
            reasoning = "Positive feedback detected"
        elif any(word in full_text for word in ['hello', 'hi', 'help', 'information', 'question']):
            category = "Service Inquiry"
            priority = "Medium"
            sentiment = "Neutral"
            reasoning = "General inquiry detected"
        else:
            category = "Other"
            priority = "Low"
            sentiment = "Neutral"
            reasoning = "General communication"
        
        return {
            "category": category,
            "priority": priority,
            "sentiment": sentiment,
            "confidence": 0.75,
            "reasoning": reasoning,
            "suggested_reply": self._generate_smart_reply(category, subject, body)
        }
    
    def _generate_smart_reply(self, category, subject, body):
        """Generate context-aware professional replies"""
        if category == "Complaint":
            return f"""Dear Customer,

We sincerely apologize for the disappointing experience you've had with: "{subject}".

This is completely unacceptable and we take full responsibility. Our senior support team has been notified and will contact you within 1 hour to resolve this matter immediately.

We are committed to making this right and restoring your confidence in our service.

Sincerely,
Customer Relations Manager"""

        elif category == "Feedback":
            return f"""Dear Customer,

Thank you so much for your wonderful feedback about: "{subject}"!

We're absolutely thrilled to hear about your positive experience! Your kind words have been shared with our entire team - this truly makes our day!

We look forward to continuing to provide you with outstanding service.

Warmest regards,
Customer Experience Team"""

        elif category == "Service Inquiry":
            return f"""Dear Customer,

Thank you for your inquiry: "{subject}".

We've received your message and our support team will get back to you within 1-2 business hours with the information you need.

In the meantime, feel free to browse our help center for quick answers to common questions.

Best regards,
Customer Support Team"""

        else:
            return f"""Dear Customer,

Thank you for your message: "{subject}".

We have received your inquiry and our team will review it shortly. We appreciate your patience and will respond as soon as possible.

Best regards,
Customer Support Team"""

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

class EmailClassifierApp:
    def __init__(self):
        self.deepseek_ai = DeepSeekAI(st.secrets.get("DEEPSEEK_API_KEY", ""))
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
        if 'processing_email' not in st.session_state:
            st.session_state.processing_email = None
    
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
    
    def classify_with_ai(self, email):
        """Classify email using DeepSeek AI"""
        try:
            with st.spinner("🧠 AI is analyzing the email..."):
                # Use DeepSeek AI for real analysis
                ai_result = self.deepseek_ai.analyze_email(email['subject'], email['body'])
                
                result = {
                    'subject': email['subject'],
                    'from': email['from'],
                    'category': ai_result['category'],
                    'priority': ai_result['priority'],
                    'sentiment': ai_result['sentiment'],
                    'confidence': ai_result['confidence'],
                    'reply': ai_result['suggested_reply'],
                    'timestamp': datetime.now().isoformat(),
                    'body_preview': email['body'][:100] + '...' if len(email['body']) > 100 else email['body'],
                    'ai_reasoning': ai_result['reasoning'],
                    'key_issues': ['AI analyzed'],
                    'model_used': 'DeepSeek AI 🤖',
                    'email_id': email['id'],
                    'original_date': email['date']
                }
                
                return result
                
        except Exception as e:
            st.error(f"❌ AI classification failed: {str(e)}")
            return None
    
    def classify_inbox_emails(self):
        """Classify all unread inbox emails with AI"""
        if not st.session_state.inbox_emails:
            st.warning("No emails to classify. Refresh inbox first.")
            return
        
        try:
            with st.spinner(f"🤖 AI is analyzing {len(st.session_state.inbox_emails)} emails..."):
                new_classifications = 0
                
                for email in st.session_state.inbox_emails:
                    # Check if email already classified
                    if not any(c.get('email_id') == email['id'] for c in st.session_state.classifications):
                        result = self.classify_with_ai(email)
                        if result:
                            st.session_state.classifications.append(result)
                            new_classifications += 1
                
                if new_classifications > 0:
                    st.success(f"✅ AI classified {new_classifications} new emails!")
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
            if st.button("🔄 Refresh", type="secondary", key="refresh_inbox_main"):
                self.fetch_inbox_emails()
        
        with col3:
            if st.button("🤖 Classify All with AI", type="primary", key="classify_all_main"):
                self.classify_inbox_emails()
        
        # Display inbox emails
        if not st.session_state.inbox_emails:
            st.info("📭 No unread emails found. Click 'Refresh Inbox' to check.")
            return
        
        st.write(f"**Found {len(st.session_state.inbox_emails)} unread emails:**")
        
        # Display each email
        for i, email in enumerate(st.session_state.inbox_emails):
            # Check if email is already classified
            is_classified = any(c.get('email_id') == email['id'] for c in st.session_state.classifications)
            classification = next((c for c in st.session_state.classifications if c.get('email_id') == email['id']), None)
            
            # Create a container for each email
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # Email subject and sender
                    st.write(f"**{email['subject']}**")
                    st.write(f"From: {email['from']} • {email['date']}")
                    
                    # Email snippet
                    st.write(f"_{email['snippet']}_")
                
                with col2:
                    # Classification status
                    if is_classified:
                        st.success(f"✅ {classification['category']}")
                        st.write(f"Priority: {classification['priority']}")
                        st.write(f"Confidence: {classification['confidence']:.0%}")
                    else:
                        st.info("📧 Unread")
                        # Classify button for this specific email
                        if st.button(f"Analyze with AI", key=f"classify_{i}"):
                            result = self.classify_with_ai(email)
                            if result:
                                st.session_state.classifications.append(result)
                                st.success(f"✅ Classified as: {result['category']}")
                                st.rerun()
                
                # Show AI reasoning if classified
                if is_classified:
                    with st.expander("🤖 AI Analysis Details"):
                        st.write(f"**Reasoning:** {classification['ai_reasoning']}")
                        st.write(f"**Sentiment:** {classification['sentiment']}")
                        st.write(f"**Suggested Reply:**")
                        st.write(classification['reply'])
                
                st.markdown("---")
    
    def render_classify_tab(self):
        """Render manual classification tab"""
        st.header("📧 Manual Classification")
        
        # Manual classification form
        st.subheader("Analyze Custom Email with AI")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            subject = st.text_input("Email Subject", 
                                  placeholder="Enter email subject...")
            sender = st.text_input("Sender", 
                                 placeholder="sender@example.com")
        
        with col2:
            email_body = st.text_area("Email Content", 
                                    height=150,
                                    placeholder="Paste email content here...")
        
        if st.button("🧠 Analyze with DeepSeek AI", type="primary", use_container_width=True, key="manual_classify"):
            if subject and email_body:
                with st.spinner("🧠 DeepSeek AI is analyzing your email..."):
                    email_data = {
                        'subject': subject,
                        'from': sender or 'customer@example.com',
                        'body': email_body,
                        'id': 'manual_' + str(int(time.time()))
                    }
                    result = self.classify_with_ai(email_data)
                    if result:
                        st.session_state.classifications.append(result)
                        self.display_classification_result(result)
            else:
                st.warning("Please enter subject and email content")
    
    def display_classification_result(self, result):
        """Display classification result"""
        with st.container():
            st.markdown("---")
            
            st.success("🎯 AI Analysis Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Category", result['category'])
            with col2:
                st.metric("Priority", result['priority'])
            with col3:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            
            # AI Reasoning
            with st.expander("🤔 AI Reasoning", expanded=True):
                st.write(result['ai_reasoning'])
            
            # AI Response
            with st.expander("📋 AI Suggested Response", expanded=True):
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
            ai_count = len([c for c in st.session_state.classifications if 'DeepSeek' in c.get('model_used', '')])
            st.metric("AI Analyzed", ai_count)
        
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
        st.subheader("Recent AI Analyses")
        for result in reversed(st.session_state.classifications[-5:]):
            self.display_email_card(result)
    
    def display_email_card(self, result):
        """Display email classification card"""
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.write(f"**{result['subject']}**")
                st.write(f"From: {result.get('from', 'Unknown')} • {result.get('model_used', 'Unknown')}")
                st.write(f"**{result['category']}** • {result['priority']} priority • {result['confidence']:.1%} confidence")
                
                with st.expander("View AI Response"):
                    st.write(result['reply'])
            
            with col2:
                priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                st.write(f"{priority_color.get(result['priority'], '⚪')} {result['priority']}")
            
            st.markdown("---")
    
    def run(self):
        """Main application runner"""
        st.title("🤖 AI Email Classifier with DeepSeek")
        
        # Check if DeepSeek API key is available
        if not st.secrets.get("DEEPSEEK_API_KEY"):
            st.error("❌ DeepSeek API key not found in secrets. Please add DEEPSEEK_API_KEY to your secrets.toml")
            return
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["📥 Gmail Inbox", "📧 Analyze Email", "📊 Analytics"])
        
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
