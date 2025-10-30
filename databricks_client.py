# databricks_client.py - Databricks API Client
import streamlit as st
import requests
import json
import base64
from typing import List, Dict, Optional

class DatabricksClient:
    def __init__(self):
        self.databricks_host = None
        self.databricks_token = None
        self.deepseek_api_key = None
        self.gmail_token = None
        
    def initialize(self) -> bool:
        """Initialize with Streamlit secrets"""
        try:
            self.databricks_host = st.secrets.get("DATABRICKS_HOST")
            self.databricks_token = st.secrets.get("DATABRICKS_TOKEN") 
            self.deepseek_api_key = st.secrets.get("DEEPSEEK_API_KEY")
            
            if not all([self.databricks_host, self.databricks_token, self.deepseek_api_key]):
                st.error("Missing required secrets. Check DATABRICKS_HOST, DATABRICKS_TOKEN, DEEPSEEK_API_KEY")
                return False
                
            return True
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            return False
    
    def _make_databricks_request(self, endpoint: str, method: str = "POST", data: dict = None) -> Optional[dict]:
        """Make authenticated request to Databricks endpoint"""
        try:
            url = f"{self.databricks_host}/api/2.0/{endpoint}"
            headers = {
                "Authorization": f"Bearer {self.databricks_token}",
                "Content-Type": "application/json"
            }
            
            if method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=30)
            else:
                response = requests.get(url, headers=headers, timeout=30)
                
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Databricks API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
            return None
    
    def connect_gmail(self, gmail_token: str) -> bool:
        """Connect to Gmail using provided token"""
        try:
            self.gmail_token = gmail_token
            # Test Gmail connection by fetching profile
            result = self._make_databricks_request(
                "gmail/connect", 
                data={"gmail_token": gmail_token}
            )
            return result and result.get("success", False)
        except Exception as e:
            st.error(f"Gmail connection failed: {str(e)}")
            return False
    
    def get_unread_emails(self, max_results: int = 10) -> List[Dict]:
        """Get unread emails from Gmail"""
        try:
            result = self._make_databricks_request(
                "gmail/emails",
                data={
                    "gmail_token": self.gmail_token,
                    "max_results": max_results,
                    "label": "INBOX",
                    "query": "is:unread"
                }
            )
            return result.get("emails", []) if result else []
        except Exception as e:
            st.error(f"Failed to fetch emails: {str(e)}")
            return []
    
    def classify_email(self, email_data: Dict) -> Optional[Dict]:
        """Classify email using AI"""
        try:
            result = self._make_databricks_request(
                "email/classify",
                data={
                    "email_data": email_data,
                    "deepseek_api_key": self.deepseek_api_key
                }
            )
            return result if result else None
        except Exception as e:
            st.error(f"Classification failed: {str(e)}")
            return None
    
    def test_connection(self) -> bool:
        """Test Databricks connection"""
        try:
            result = self._make_databricks_request("health", method="GET")
            return result and result.get("status") == "healthy"
        except:
            return False
    
    def test_deepseek(self) -> bool:
        """Test DeepSeek API connection"""
        try:
            result = self._make_databricks_request(
                "test/deepseek",
                data={"deepseek_api_key": self.deepseek_api_key}
            )
            return result and result.get("success", False)
        except:
            return False
