#!/usr/bin/env python3
"""
B2B Data QA Copilot - Web Edition
Offline Localhost Application
Run: python b2b_qa_app.py
Then open: http://localhost:5000
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import re
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

# Flask imports
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_socketio import SocketIO
import threading
import eventlet
eventlet.monkey_patch()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    SECRET_KEY = 'b2b-qa-copilot-local-offline-key'
    UPLOAD_FOLDER = './uploads'
    RESULTS_FOLDER = './results'
    ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx', 'xls'}
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    
    @staticmethod
    def init_dirs():
        """Create necessary directories"""
        for folder in [Config.UPLOAD_FOLDER, Config.RESULTS_FOLDER]:
            os.makedirs(folder, exist_ok=True)

Config.init_dirs()

# ============================================================================
# DATA MODELS
# ============================================================================

class IssueType(Enum):
    MISSING = "Missing Field"
    INVALID_EMAIL = "Invalid Email"
    INVALID_DOMAIN = "Invalid Domain"
    INVALID_PHONE = "Invalid Phone"
    INVALID_URL = "Invalid URL"
    INDUSTRY_NORMALIZATION = "Industry Normalization"
    TITLE_NORMALIZATION = "Job Title Normalization"
    COUNTRY_NORMALIZATION = "Country Normalization"
    DOMAIN_MISMATCH = "Domain Mismatch"
    DUPLICATE = "Duplicate Record"
    SUSPICIOUS_VALUE = "Suspicious Value"

class ConfidenceLevel(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class DataType(Enum):
    COMPANY = "Company"
    PEOPLE = "People"

@dataclass
class QAIssue:
    record_id: str
    field: str
    issue_type: IssueType
    original_value: str
    suggested_value: str
    reason: str
    confidence: float
    confidence_level: ConfidenceLevel
    
    def to_dict(self):
        return {
            'record_id': self.record_id,
            'field': self.field,
            'issue_type': self.issue_type.value,
            'original_value': str(self.original_value),
            'suggested_value': str(self.suggested_value),
            'reason': self.reason,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value
        }

@dataclass
class DuplicateCluster:
    cluster_id: str
    record_ids: List[str]
    similarity_score: float
    merge_recommendation: str
    
    def to_dict(self):
        return {
            'cluster_id': self.cluster_id,
            'record_count': len(self.record_ids),
            'record_ids': self.record_ids,
            'similarity_score': self.similarity_score,
            'recommendation': self.merge_recommendation
        }

# Standard taxonomies
INDUSTRY_TAXONOMY = {
    'information technology': ['it', 'tech', 'software', 'hardware', 'saas', 'cloud', 'technology'],
    'healthcare': ['health', 'medical', 'pharma', 'biotech', 'hospital', 'clinic'],
    'finance': ['banking', 'financial', 'fintech', 'insurance', 'investment', 'bank'],
    'manufacturing': ['industrial', 'factory', 'production', 'manufacture', 'factory'],
    'retail': ['ecommerce', 'shop', 'store', 'retail', 'shopping'],
    'consulting': ['consultancy', 'professional services', 'consulting', 'advisory'],
    'education': ['edtech', 'school', 'university', 'college', 'education', 'learning'],
    'real estate': ['property', 'construction', 'real estate', 'housing', 'property'],
    'transportation': ['logistics', 'shipping', 'delivery', 'transport', 'logistic'],
    'energy': ['utilities', 'oil', 'gas', 'renewable', 'energy', 'power'],
    'telecommunications': ['telecom', 'communications', 'wireless', 'broadband'],
    'media': ['entertainment', 'publishing', 'media', 'news', 'broadcasting'],
    'automotive': ['auto', 'car', 'vehicle', 'automobile', 'automotive'],
    'aerospace': ['aviation', 'aircraft', 'space', 'aerospace', 'defense'],
    'pharmaceutical': ['pharma', 'drug', 'medicine', 'pharmaceutical', 'biopharma']
}

JOB_FUNCTIONS = {
    'Sales': ['sales', 'account executive', 'business development', 'account manager', 'ae', 'bdr', 'sdr'],
    'Marketing': ['marketing', 'demand gen', 'growth', 'brand', 'content', 'digital marketing', 'campaign'],
    'Engineering': ['engineer', 'developer', 'architect', 'technical', 'software', 'dev', 'engineering'],
    'IT': ['it', 'support', 'system admin', 'network', 'infrastructure', 'sysadmin', 'helpdesk'],
    'HR': ['hr', 'human resources', 'talent', 'recruitment', 'people', 'recruiter', 'hiring'],
    'Finance': ['finance', 'accounting', 'cfo', 'controller', 'financial', 'accountant', 'bookkeeping'],
    'Operations': ['operations', 'ops', 'supply chain', 'logistics', 'operational', 'facilities'],
    'Product': ['product', 'pm', 'product manager', 'product owner', 'product management'],
    'Leadership': ['ceo', 'cto', 'cmo', 'coo', 'founder', 'director', 'vp', 'head of', 'president', 'chief'],
    'Customer Success': ['customer success', 'cs', 'account management', 'customer support', 'support'],
    'Research & Development': ['research', 'r&d', 'development', 'scientist', 'researcher', 'lab'],
    'Legal': ['legal', 'lawyer', 'counsel', 'attorney', 'compliance', 'regulatory']
}

COUNTRY_STANDARDIZATION = {
    'usa': 'United States',
    'us': 'United States',
    'u.s.': 'United States',
    'united states': 'United States',
    'america': 'United States',
    'uk': 'United Kingdom',
    'u.k.': 'United Kingdom',
    'united kingdom': 'United Kingdom',
    'great britain': 'United Kingdom',
    'gb': 'United Kingdom',
    'canada': 'Canada',
    'ca': 'Canada',
    'australia': 'Australia',
    'au': 'Australia',
    'germany': 'Germany',
    'de': 'Germany',
    'france': 'France',
    'fr': 'France',
    'india': 'India',
    'in': 'India',
    'china': 'China',
    'cn': 'China',
    'japan': 'Japan',
    'jp': 'Japan',
    'brazil': 'Brazil',
    'br': 'Brazil',
    'mexico': 'Mexico',
    'mx': 'Mexico',
    'spain': 'Spain',
    'es': 'Spain',
    'italy': 'Italy',
    'it': 'Italy',
    'netherlands': 'Netherlands',
    'nl': 'Netherlands',
    'switzerland': 'Switzerland',
    'ch': 'Switzerland',
    'singapore': 'Singapore',
    'sg': 'Singapore'
}

# ============================================================================
# CORE QA ENGINE
# ============================================================================

class B2BDataQACopilot:
    def __init__(self):
        self.data = None
        self.data_type = None
        self.issues = []
        self.duplicate_clusters = []
        self.quality_score = 0
        self.metrics = {}
        self.confidence_threshold = 70
        self.auto_fix_enabled = False
        self.progress_callback = None
        self.session_id = None
        
    def set_progress_callback(self, callback):
        """Set callback for progress updates"""
        self.progress_callback = callback
    
    def update_progress(self, stage, progress, message=""):
        """Update progress through callback"""
        if self.progress_callback:
            self.progress_callback({
                'stage': stage,
                'progress': progress,
                'message': message,
                'session_id': self.session_id
            })
    
    def load_data(self, file_path: str, data_type: DataType) -> bool:
        """Load and validate uploaded data"""
        try:
            self.update_progress('loading', 10, "Loading data file...")
            
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            else:
                return False
                
            self.data_type = data_type
            
            # Clean column names
            self.data.columns = self.data.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Add record IDs
            self.data['record_id'] = [f"{data_type.value[:3].lower()}_{i:06d}" for i in range(len(self.data))]
            
            # Initialize metrics
            self.metrics = {
                'total_records': len(self.data),
                'issues_found': 0,
                'completeness': 0,
                'validity': 0,
                'standardization_rate': 0,
                'duplicate_rate': 0,
                'quality_score': 0
            }
            
            self.update_progress('loading', 100, "Data loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def run_qa_pipeline(self):
        """Execute complete QA pipeline"""
        if self.data is None:
            return
            
        try:
            self.issues = []
            self.duplicate_clusters = []
            
            # Step 1: Validate required fields
            self.update_progress('validation', 20, "Validating required fields...")
            self._validate_required_fields()
            
            # Step 2: Validate formats
            self.update_progress('validation', 40, "Validating data formats...")
            self._validate_formats()
            
            # Step 3: Normalize data
            self.update_progress('normalization', 60, "Normalizing data...")
            self._normalize_data()
            
            # Step 4: Detect duplicates
            self.update_progress('deduplication', 80, "Detecting duplicates...")
            self._detect_duplicates()
            
            # Step 5: Calculate scores
            self.update_progress('scoring', 90, "Calculating quality scores...")
            self._calculate_quality_score()
            
            # Apply auto-fixes if enabled
            if self.auto_fix_enabled:
                self.update_progress('fixing', 95, "Applying automatic fixes...")
                self._apply_auto_fixes()
            
            self.update_progress('complete', 100, "QA pipeline completed!")
            
        except Exception as e:
            print(f"Error in QA pipeline: {e}")
            self.update_progress('error', 0, f"Error: {str(e)}")
    
    def _validate_required_fields(self):
        """Check for missing required fields"""
        required_fields = {
            DataType.COMPANY: ['company_name', 'domain'],
            DataType.PEOPLE: ['full_name', 'email', 'company_name']
        }
        
        fields = required_fields.get(self.data_type, [])
        for field in fields:
            if field in self.data.columns:
                missing_mask = self.data[field].isna() | (self.data[field].astype(str) == '')
                missing_records = self.data[missing_mask]
                
                for idx, record in missing_records.iterrows():
                    self.issues.append(QAIssue(
                        record_id=record['record_id'],
                        field=field,
                        issue_type=IssueType.MISSING,
                        original_value='',
                        suggested_value='[REQUIRED]',
                        reason=f"Required field '{field}' is missing",
                        confidence=95.0,
                        confidence_level=ConfidenceLevel.HIGH
                    ))
    
    def _validate_formats(self):
        """Validate email, phone, domain, URL formats"""
        # Email validation
        if 'email' in self.data.columns:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            for idx, record in self.data.iterrows():
                email = str(record.get('email', ''))
                if email and email.lower() != 'nan':
                    if not re.match(email_pattern, email.lower()):
                        suggestion = self._suggest_email_correction(email, record.get('domain', ''))
                        self.issues.append(QAIssue(
                            record_id=record['record_id'],
                            field='email',
                            issue_type=IssueType.INVALID_EMAIL,
                            original_value=email,
                            suggested_value=suggestion,
                            reason="Email format is invalid",
                            confidence=90.0 if suggestion != email else 85.0,
                            confidence_level=ConfidenceLevel.HIGH
                        ))
        
        # Domain validation
        if 'domain' in self.data.columns:
            domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+$'
            for idx, record in self.data.iterrows():
                domain = str(record.get('domain', ''))
                if domain and domain.lower() != 'nan':
                    domain_lower = domain.lower()
                    if not re.match(domain_pattern, domain_lower):
                        suggestion = self._suggest_domain_correction(domain_lower)
                        self.issues.append(QAIssue(
                            record_id=record['record_id'],
                            field='domain',
                            issue_type=IssueType.INVALID_DOMAIN,
                            original_value=domain,
                            suggested_value=suggestion,
                            reason="Domain format is invalid",
                            confidence=80.0 if suggestion != domain_lower else 75.0,
                            confidence_level=ConfidenceLevel.MEDIUM
                        ))
        
        # Phone validation
        if 'phone' in self.data.columns:
            for idx, record in self.data.iterrows():
                phone = str(record.get('phone', ''))
                if phone and phone.lower() != 'nan':
                    cleaned = re.sub(r'[\s\-\(\)\.\+]', '', phone)
                    if not cleaned.isdigit() or len(cleaned) < 7:
                        self.issues.append(QAIssue(
                            record_id=record['record_id'],
                            field='phone',
                            issue_type=IssueType.INVALID_PHONE,
                            original_value=phone,
                            suggested_value=cleaned if cleaned.isdigit() and len(cleaned) >= 7 else '[INVALID]',
                            reason="Phone number format is invalid",
                            confidence=85.0,
                            confidence_level=ConfidenceLevel.HIGH
                        ))
    
    def _suggest_email_correction(self, email: str, domain: str) -> str:
        """Suggest email corrections"""
        if '@' not in email:
            return email
            
        local_part, current_domain = email.split('@', 1)
        
        # Correct common typos
        domain_corrections = {
            'gmail.com': ['gmail.co', 'gmil.com', 'gmal.com', 'gmail.cm', 'gmail.con'],
            'outlook.com': ['outlok.com', 'outlook.co', 'outlook.cm'],
            'yahoo.com': ['yaho.com', 'yahoo.co', 'yahoo.cm', 'yhaoo.com'],
            'hotmail.com': ['hotmal.com', 'hotmail.co', 'hotmail.cm'],
            'company.com': ['company.co', 'company.cm', 'comany.com']
        }
        
        for correct, typos in domain_corrections.items():
            if current_domain in typos:
                return f"{local_part}@{correct}"
        
        # Use company domain if available
        if domain and str(domain).lower() != 'nan':
            clean_domain = str(domain).lower().replace('www.', '').replace('http://', '').replace('https://', '').split('/')[0]
            if clean_domain and '.' in clean_domain:
                return f"{local_part}@{clean_domain}"
        
        return email
    
    def _suggest_domain_correction(self, domain: str) -> str:
        """Suggest domain corrections"""
        # Common typos
        corrections = {
            'gooogle.com': 'google.com',
            'gogle.com': 'google.com',
            'google.co': 'google.com',
            'facebok.com': 'facebook.com',
            'facebook.co': 'facebook.com',
            'linkdin.com': 'linkedin.com',
            'linkedin.co': 'linkedin.com',
            'microsoft.co': 'microsoft.com',
            'amazn.com': 'amazon.com',
            'amazon.co': 'amazon.com',
            'appl.com': 'apple.com',
            'apple.co': 'apple.com',
            'twiter.com': 'twitter.com',
            'twitter.co': 'twitter.com'
        }
        
        for typo, correct in corrections.items():
            if typo in domain:
                return correct
        
        # Fix missing TLD
        if '.' not in domain:
            if domain.endswith('co'):
                return domain + 'm'
            elif domain.endswith('cm'):
                return domain[:-2] + 'com'
            else:
                return domain + '.com'
        
        return domain
    
    def _normalize_data(self):
        """Normalize industry, job titles, countries"""
        # Industry normalization
        if 'industry' in self.data.columns:
            for idx, record in self.data.iterrows():
                industry = str(record.get('industry', '')).lower()
                if industry and industry != 'nan':
                    standardized = self._standardize_industry(industry)
                    if standardized != industry:
                        self.issues.append(QAIssue(
                            record_id=record['record_id'],
                            field='industry',
                            issue_type=IssueType.INDUSTRY_NORMALIZATION,
                            original_value=record.get('industry', ''),
                            suggested_value=standardized.title(),
                            reason=f"Standardized to industry taxonomy",
                            confidence=self._calculate_similarity(industry, standardized) * 100,
                            confidence_level=ConfidenceLevel.MEDIUM
                        ))
        
        # Job title normalization
        if 'job_title' in self.data.columns:
            for idx, record in self.data.iterrows():
                job_title = str(record.get('job_title', '')).lower()
                if job_title and job_title != 'nan':
                    function, confidence = self._map_job_to_function(job_title)
                    if function:
                        self.issues.append(QAIssue(
                            record_id=record['record_id'],
                            field='job_title',
                            issue_type=IssueType.TITLE_NORMALIZATION,
                            original_value=record.get('job_title', ''),
                            suggested_value=f"{function}",
                            reason=f"Mapped to {function} function",
                            confidence=confidence,
                            confidence_level=ConfidenceLevel.HIGH if confidence > 85 else ConfidenceLevel.MEDIUM
                        ))
        
        # Country normalization
        if 'country' in self.data.columns:
            for idx, record in self.data.iterrows():
                country = str(record.get('country', '')).lower()
                if country and country != 'nan':
                    standardized = COUNTRY_STANDARDIZATION.get(country, country.title())
                    if standardized.lower() != country:
                        self.issues.append(QAIssue(
                            record_id=record['record_id'],
                            field='country',
                            issue_type=IssueType.COUNTRY_NORMALIZATION,
                            original_value=record.get('country', ''),
                            suggested_value=standardized,
                            reason="Standardized country name",
                            confidence=90.0,
                            confidence_level=ConfidenceLevel.HIGH
                        ))
    
    def _standardize_industry(self, industry: str) -> str:
        """Standardize industry name"""
        industry_lower = industry.lower()
        
        # Direct matches
        for standard, variations in INDUSTRY_TAXONOMY.items():
            if industry_lower == standard:
                return standard
            for variation in variations:
                if variation == industry_lower:
                    return standard
        
        # Partial matches
        for standard, variations in INDUSTRY_TAXONOMY.items():
            if standard in industry_lower:
                return standard
            for variation in variations:
                if variation in industry_lower:
                    return standard
        
        # Word-based matching
        words = set(industry_lower.split())
        best_match = industry_lower
        best_score = 0
        
        for standard, variations in INDUSTRY_TAXONOMY.items():
            standard_words = set(standard.split())
            variation_words = set()
            for v in variations:
                variation_words.update(v.split())
            
            all_words = standard_words.union(variation_words)
            common = words.intersection(all_words)
            score = len(common) / max(len(words), 1)
            
            if score > best_score and score > 0.3:
                best_score = score
                best_match = standard
        
        return best_match
    
    def _map_job_to_function(self, job_title: str) -> Tuple[str, float]:
        """Map job title to standardized function"""
        job_lower = job_title.lower()
        
        best_function = ""
        best_confidence = 0
        
        # Check for exact matches first
        for function, keywords in JOB_FUNCTIONS.items():
            function_lower = function.lower()
            
            # Exact function name in title
            if function_lower in job_lower:
                return function, 95.0
            
            # Check keywords
            for keyword in keywords:
                if keyword in job_lower:
                    confidence = min(90.0 + (len(keyword) * 3), 98.0)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_function = function
        
        # Check for seniority indicators
        if not best_function:
            seniority_terms = ['senior', 'lead', 'principal', 'head of', 'director', 'vp', 'c-level', 'chief', 'president']
            for term in seniority_terms:
                if term in job_lower:
                    return "Leadership", 85.0
        
        return best_function, best_confidence
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity"""
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_duplicates(self):
        """Detect duplicate records"""
        if self.data_type == DataType.COMPANY and 'company_name' in self.data.columns:
            self._detect_company_duplicates()
        elif self.data_type == DataType.PEOPLE:
            self._detect_people_duplicates()
    
    def _detect_company_duplicates(self):
        """Detect duplicate companies"""
        if 'company_name' not in self.data.columns:
            return
            
        # Normalize company names
        normalized_names = self.data['company_name'].fillna('').astype(str).str.lower().str.strip()
        name_groups = self.data.groupby(normalized_names)
        
        for name, group in name_groups:
            if len(group) > 1 and name:  # Only if we have a name
                cluster_id = f"COMPANY_CLUSTER_{len(self.duplicate_clusters)+1:03d}"
                record_ids = group['record_id'].tolist()
                
                # Calculate similarity based on domain
                similarities = []
                if 'domain' in group.columns:
                    domains = group['domain'].fillna('').astype(str).str.lower()
                    for i in range(len(group)):
                        for j in range(i+1, len(group)):
                            sim = self._calculate_similarity(
                                domains.iloc[i],
                                domains.iloc[j]
                            )
                            similarities.append(sim)
                
                avg_similarity = np.mean(similarities) * 100 if similarities else 75.0
                
                self.duplicate_clusters.append(DuplicateCluster(
                    cluster_id=cluster_id,
                    record_ids=record_ids,
                    similarity_score=avg_similarity,
                    merge_recommendation=f"Keep record with most complete data from cluster {cluster_id}"
                ))
                
                for record_id in record_ids:
                    self.issues.append(QAIssue(
                        record_id=record_id,
                        field='company_name',
                        issue_type=IssueType.DUPLICATE,
                        original_value=name,
                        suggested_value=f"[DUPLICATE - {cluster_id}]",
                        reason=f"Potential duplicate with {len(group)-1} other records",
                        confidence=avg_similarity,
                        confidence_level=ConfidenceLevel.HIGH if avg_similarity > 80 else ConfidenceLevel.MEDIUM
                    ))
    
    def _detect_people_duplicates(self):
        """Detect duplicate people records"""
        # Group by email
        if 'email' in self.data.columns:
            emails = self.data['email'].fillna('').astype(str).str.lower().str.strip()
            email_groups = self.data.groupby(emails)
            
            for email, group in email_groups:
                if email and len(group) > 1:
                    self._create_people_duplicate_cluster(group, 'email')
        
        # Group by name + company
        if 'full_name' in self.data.columns and 'company_name' in self.data.columns:
            names = self.data['full_name'].fillna('').astype(str).str.lower().str.strip()
            companies = self.data['company_name'].fillna('').astype(str).str.lower().str.strip()
            key = names + '_' + companies
            name_groups = self.data.groupby(key)
            
            for _, group in name_groups:
                if len(group) > 1:
                    self._create_people_duplicate_cluster(group, 'name+company')
    
    def _create_people_duplicate_cluster(self, group: pd.DataFrame, match_type: str):
        """Create duplicate cluster for people"""
        cluster_id = f"PEOPLE_CLUSTER_{len(self.duplicate_clusters)+1:03d}"
        record_ids = group['record_id'].tolist()
        
        self.duplicate_clusters.append(DuplicateCluster(
            cluster_id=cluster_id,
            record_ids=record_ids,
            similarity_score=85.0 if match_type == 'email' else 75.0,
            merge_recommendation=f"Merge {len(group)} records matched by {match_type}"
        ))
        
        for record_id in record_ids:
            self.issues.append(QAIssue(
                record_id=record_id,
                field='full_name' if 'full_name' in group.columns else 'email',
                issue_type=IssueType.DUPLICATE,
                original_value=str(group[group['record_id'] == record_id].iloc[0].get('full_name', 'email')),
                suggested_value=f"[DUPLICATE - {cluster_id}]",
                reason=f"Potential duplicate ({match_type} match)",
                confidence=85.0 if match_type == 'email' else 75.0,
                confidence_level=ConfidenceLevel.HIGH if match_type == 'email' else ConfidenceLevel.MEDIUM
            ))
    
    def _calculate_quality_score(self):
        """Calculate overall data quality score"""
        if self.data is None or len(self.data) == 0:
            self.quality_score = 0
            return
        
        # Calculate completeness
        required_fields = {
            DataType.COMPANY: ['company_name', 'domain'],
            DataType.PEOPLE: ['full_name', 'email', 'company_name']
        }
        
        fields = required_fields.get(self.data_type, [])
        completeness_scores = []
        
        for field in fields:
            if field in self.data.columns:
                filled = self.data[field].notna().sum() + (self.data[field].astype(str) != '').sum()
                completeness_scores.append(filled / len(self.data))
        
        completeness = np.mean(completeness_scores) * 100 if completeness_scores else 0
        
        # Calculate validity
        high_confidence_issues = len([i for i in self.issues if i.confidence > self.confidence_threshold])
        issue_ratio = high_confidence_issues / (len(self.data) * 3)  # Normalize by estimated max issues
        validity = 100 - min(100, issue_ratio * 100)
        
        # Calculate standardization rate
        standardization_issues = len([i for i in self.issues if 'normalization' in i.issue_type.value.lower()])
        standardization_rate = 100 - min(100, (standardization_issues / max(1, len(self.data))) * 50)
        
        # Calculate duplicate rate
        duplicate_records = len(set([i.record_id for i in self.issues if i.issue_type == IssueType.DUPLICATE]))
        duplicate_rate = 100 - min(100, (duplicate_records / len(self.data)) * 100) if len(self.data) > 0 else 100
        
        # Weighted quality score
        weights = {
            'completeness': 0.30,
            'validity': 0.40,
            'standardization': 0.20,
            'duplicate_free': 0.10
        }
        
        self.quality_score = round(
            completeness * weights['completeness'] +
            validity * weights['validity'] +
            standardization_rate * weights['standardization'] +
            duplicate_rate * weights['duplicate_free']
        , 1)
        
        self.metrics.update({
            'issues_found': len(self.issues),
            'completeness': round(completeness, 1),
            'validity': round(validity, 1),
            'standardization_rate': round(standardization_rate, 1),
            'duplicate_rate': round(duplicate_rate, 1),
            'quality_score': self.quality_score
        })
    
    def _apply_auto_fixes(self):
        """Apply automatic fixes based on issues"""
        if self.data is None:
            return
            
        fixed_data = self.data.copy()
        
        for issue in self.issues:
            if issue.confidence > self.confidence_threshold:
                if issue.field in fixed_data.columns:
                    mask = fixed_data['record_id'] == issue.record_id
                    if (issue.issue_type in [IssueType.INDUSTRY_NORMALIZATION, 
                                           IssueType.COUNTRY_NORMALIZATION,
                                           IssueType.TITLE_NORMALIZATION] or
                        'invalid' in issue.issue_type.value.lower()):
                        if not issue.suggested_value.startswith('['):
                            fixed_data.loc[mask, issue.field] = issue.suggested_value
        
        self.data = fixed_data
    
    def get_cleaned_data(self, format: str = 'csv') -> str:
        """Get cleaned data in specified format"""
        if self.data is None:
            return ""
        
        # Remove internal columns
        export_data = self.data.drop(columns=['record_id'], errors='ignore')
        
        if format.lower() == 'csv':
            return export_data.to_csv(index=False)
        elif format.lower() == 'json':
            return export_data.to_json(orient='records', indent=2, default_handler=str)
        else:
            return export_data.to_string()
    
    def generate_qa_report(self) -> Dict:
        """Generate comprehensive QA report"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_type': self.data_type.value if self.data_type else 'Unknown',
            'total_records': self.metrics.get('total_records', 0),
            'quality_score': self.metrics.get('quality_score', 0),
            'metrics': self.metrics,
            'issues_by_type': {},
            'issue_details': [issue.to_dict() for issue in self.issues[:100]],  # Limit details
            'duplicate_clusters': [cluster.to_dict() for cluster in self.duplicate_clusters]
        }
        
        # Group issues by type
        for issue in self.issues:
            issue_type = issue.issue_type.value
            report['issues_by_type'][issue_type] = report['issues_by_type'].get(issue_type, 0) + 1
        
        return report

# ============================================================================
# FLASK APPLICATION
# ============================================================================

app = Flask(__name__, static_folder='.', static_url_path='')
app.config.from_object(Config)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Session storage for multiple users
qa_sessions = {}
file_sessions = {}

# HTML Template (embedded in Python for single file)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 B2B Data QA Copilot | Offline Edition</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --bg-darker: #0a0f1a;
            --primary: #3b82f6;
            --primary-dark: #2563eb;
            --secondary: #8b5cf6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --border: #334155;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, var(--bg-darker), var(--bg-dark));
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 30px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .logo h1 {
            font-size: 28px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .offline-badge {
            background: var(--success);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        /* Navigation */
        .nav-tabs {
            display: flex;
            gap: 5px;
            background: var(--bg-card);
            padding: 5px;
            border-radius: 12px;
            margin-bottom: 30px;
            border: 1px solid var(--border);
        }
        
        .nav-tab {
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 500;
        }
        
        .nav-tab:hover {
            background: rgba(255, 255, 255, 0.05);
        }
        
        .nav-tab.active {
            background: var(--primary);
            color: white;
        }
        
        /* Cards */
        .card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 25px;
            border: 1px solid var(--border);
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        
        .card-title {
            font-size: 18px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text);
        }
        
        /* Upload Area */
        .upload-area {
            border: 3px dashed var(--border);
            border-radius: 16px;
            padding: 60px 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.02);
        }
        
        .upload-area:hover {
            border-color: var(--primary);
            background: rgba(59, 130, 246, 0.05);
        }
        
        .upload-area i {
            font-size: 64px;
            color: var(--primary);
            margin-bottom: 20px;
        }
        
        .upload-area h3 {
            font-size: 22px;
            margin-bottom: 10px;
        }
        
        .upload-area p {
            color: var(--text-muted);
            margin-bottom: 25px;
        }
        
        .btn {
            background: var(--primary);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
        }
        
        .btn-success {
            background: var(--success);
        }
        
        .btn-warning {
            background: var(--warning);
        }
        
        /* Progress */
        .progress-container {
            margin: 30px 0;
        }
        
        .progress-stages {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            position: relative;
        }
        
        .progress-stage {
            text-align: center;
            position: relative;
            z-index: 2;
            flex: 1;
        }
        
        .stage-circle {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: var(--bg-card);
            border: 2px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 10px;
            transition: all 0.3s ease;
        }
        
        .stage-circle.active {
            background: var(--primary);
            border-color: var(--primary);
            color: white;
        }
        
        .stage-circle.completed {
            background: var(--success);
            border-color: var(--success);
            color: white;
        }
        
        .progress-bar {
            height: 4px;
            background: var(--border);
            position: absolute;
            top: 20px;
            left: 5%;
            right: 5%;
            z-index: 1;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            width: 0%;
            transition: width 0.5s ease;
            border-radius: 2px;
        }
        
        /* Metrics */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.8));
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border);
        }
        
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .quality-score {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Tables */
        .table-container {
            overflow-x: auto;
            border-radius: 12px;
            border: 1px solid var(--border);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th {
            background: rgba(30, 41, 59, 0.8);
            padding: 15px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid var(--border);
        }
        
        td {
            padding: 12px 15px;
            border-bottom: 1px solid var(--border);
        }
        
        tr:hover {
            background: rgba(255, 255, 255, 0.05);
        }
        
        .issue-high { color: var(--success); }
        .issue-medium { color: var(--warning); }
        .issue-low { color: var(--danger); }
        
        /* Forms */
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        select, input[type="range"] {
            width: 100%;
            padding: 12px;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-size: 16px;
        }
        
        .checkbox-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .checkbox-label {
            display: flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
        }
        
        /* Animations */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .gradient-bg {
            background: linear-gradient(-45deg, #3b82f6, #8b5cf6, #ec4899, #3b82f6);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .header {
                flex-direction: column;
                gap: 15px;
                text-align: center;
            }
            
            .nav-tabs {
                overflow-x: auto;
                flex-wrap: nowrap;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="logo">
                <i class="fas fa-brain fa-2x" style="color: #8b5cf6;"></i>
                <h1>B2B Data QA Copilot</h1>
            </div>
            <div class="offline-badge">
                <i class="fas fa-wifi-slash"></i>
                OFFLINE MODE
            </div>
        </div>
        
        <!-- Navigation Tabs -->
        <div class="nav-tabs" id="navTabs">
            <div class="nav-tab active" onclick="switchTab('upload')">
                <i class="fas fa-upload"></i>
                Upload Data
            </div>
            <div class="nav-tab" onclick="switchTab('qa')">
                <i class="fas fa-cog"></i>
                QA Configuration
            </div>
            <div class="nav-tab" onclick="switchTab('results')">
                <i class="fas fa-chart-bar"></i>
                Results & Review
            </div>
            <div class="nav-tab" onclick="switchTab('report')">
                <i class="fas fa-file-alt"></i>
                Report & Export
            </div>
        </div>
        
        <!-- Upload Tab -->
        <div id="uploadTab" class="tab-content">
            <div class="card">
                <h2 class="card-title"><i class="fas fa-file-upload"></i> Upload Your B2B Data</h2>
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <i class="fas fa-cloud-upload-alt"></i>
                    <h3>Drag & Drop or Click to Upload</h3>
                    <p>Supports CSV, JSON, Excel files</p>
                    <button class="btn">
                        <i class="fas fa-folder-open"></i>
                        Browse Files
                    </button>
                </div>
                <input type="file" id="fileInput" accept=".csv,.json,.xlsx,.xls" style="display: none;" onchange="handleFileUpload(this.files)">
                
                <div class="form-group" style="margin-top: 30px;">
                    <label>Data Type</label>
                    <select id="dataType">
                        <option value="company">Company Data</option>
                        <option value="people">People Data</option>
                    </select>
                </div>
                
                <!-- File Preview -->
                <div id="filePreview" style="display: none;">
                    <h3 class="card-title"><i class="fas fa-eye"></i> File Preview</h3>
                    <div class="table-container">
                        <table id="previewTable">
                            <thead>
                                <tr>
                                    <th>Column</th>
                                    <th>Sample Value</th>
                                    <th>Type</th>
                                </tr>
                            </thead>
                            <tbody id="previewBody">
                                <!-- Filled by JavaScript -->
                            </tbody>
                        </table>
                    </div>
                    <button class="btn btn-success" onclick="startQA()" style="margin-top: 20px;">
                        <i class="fas fa-play"></i>
                        Start QA Analysis
                    </button>
                </div>
            </div>
        </div>
        
        <!-- QA Configuration Tab -->
        <div id="qaTab" class="tab-content" style="display: none;">
            <div class="card">
                <h2 class="card-title"><i class="fas fa-sliders-h"></i> QA Configuration</h2>
                
                <div class="form-group">
                    <label>Confidence Threshold: <span id="confidenceValue">70%</span></label>
                    <input type="range" id="confidenceSlider" min="50" max="95" value="70" 
                           oninput="updateConfidenceValue(this.value)">
                </div>
                
                <div class="form-group">
                    <label>QA Features</label>
                    <div class="checkbox-group">
                        <label class="checkbox-label">
                            <input type="checkbox" id="autoFix" checked>
                            <span>Auto-fix high confidence issues</span>
                        </label>
                        <label class="checkbox-label">
                            <input type="checkbox" id="deduplication" checked>
                            <span>Detect duplicate records</span>
                        </label>
                        <label class="checkbox-label">
                            <input type="checkbox" id="normalization" checked>
                            <span>Normalize industries & job titles</span>
                        </label>
                    </div>
                </div>
                
                <!-- Progress Visualization -->
                <div class="progress-container" id="progressSection" style="display: none;">
                    <h3 class="card-title"><i class="fas fa-tasks"></i> QA Pipeline Progress</h3>
                    <div class="progress-stages">
                        <div class="progress-stage">
                            <div class="stage-circle" id="stage1">1</div>
                            <div>Loading</div>
                        </div>
                        <div class="progress-stage">
                            <div class="stage-circle" id="stage2">2</div>
                            <div>Validation</div>
                        </div>
                        <div class="progress-stage">
                            <div class="stage-circle" id="stage3">3</div>
                            <div>Normalization</div>
                        </div>
                        <div class="progress-stage">
                            <div class="stage-circle" id="stage4">4</div>
                            <div>Deduplication</div>
                        </div>
                        <div class="progress-stage">
                            <div class="stage-circle" id="stage5">5</div>
                            <div>Scoring</div>
                        </div>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                    
                    <div id="progressMessage" style="text-align: center; margin-top: 20px; color: var(--text-muted);">
                        Ready to start QA pipeline...
                    </div>
                </div>
                
                <button class="btn btn-success" onclick="runQAPipeline()" style="width: 100%; padding: 15px;">
                    <i class="fas fa-rocket"></i>
                    Run QA Pipeline
                </button>
            </div>
        </div>
        
        <!-- Results Tab -->
        <div id="resultsTab" class="tab-content" style="display: none;">
            <div class="card">
                <h2 class="card-title"><i class="fas fa-chart-line"></i> QA Results Dashboard</h2>
                
                <!-- Quality Score -->
                <div style="text-align: center; margin-bottom: 40px;">
                    <div class="quality-score" id="qualityScore">0</div>
                    <div style="color: var(--text-muted); font-size: 18px;">Overall Data Quality Score</div>
                </div>
                
                <!-- Metrics -->
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="totalRecords">0</div>
                        <div>Total Records</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="issuesFound">0</div>
                        <div>Issues Found</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="completeness">0%</div>
                        <div>Completeness</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="validity">0%</div>
                        <div>Validity</div>
                    </div>
                </div>
                
                <!-- Issues Table -->
                <h3 class="card-title"><i class="fas fa-exclamation-triangle"></i> Issues Found</h3>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Record ID</th>
                                <th>Field</th>
                                <th>Issue Type</th>
                                <th>Confidence</th>
                                <th>Suggested Fix</th>
                            </tr>
                        </thead>
                        <tbody id="issuesTableBody">
                            <!-- Filled by JavaScript -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Report Tab -->
        <div id="reportTab" class="tab-content" style="display: none;">
            <div class="card">
                <h2 class="card-title"><i class="fas fa-file-download"></i> Export & Reports</h2>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px;">
                    <div class="metric-card">
                        <h3><i class="fas fa-file-csv"></i> Cleaned Data</h3>
                        <p>Export your cleaned and standardized data</p>
                        <button class="btn" onclick="exportData('csv')" style="margin-top: 15px;">
                            <i class="fas fa-download"></i>
                            Export as CSV
                        </button>
                        <button class="btn" onclick="exportData('json')" style="margin-top: 10px;">
                            <i class="fas fa-download"></i>
                            Export as JSON
                        </button>
                    </div>
                    
                    <div class="metric-card">
                        <h3><i class="fas fa-chart-pie"></i> QA Report</h3>
                        <p>Detailed analysis and recommendations</p>
                        <button class="btn" onclick="exportReport()" style="margin-top: 15px;">
                            <i class="fas fa-download"></i>
                            Download Report
                        </button>
                        <button class="btn" onclick="showReport()" style="margin-top: 10px;">
                            <i class="fas fa-eye"></i>
                            View Report
                        </button>
                    </div>
                </div>
                
                <!-- Report Preview -->
                <div id="reportPreview" style="display: none;">
                    <h3 class="card-title"><i class="fas fa-file-alt"></i> QA Report Preview</h3>
                    <div style="background: var(--bg-dark); padding: 20px; border-radius: 8px; border: 1px solid var(--border);">
                        <pre id="reportContent" style="color: var(--text); white-space: pre-wrap; font-family: monospace;"></pre>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- JavaScript -->
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <script>
        let socket;
        let currentSessionId = null;
        
        // Initialize WebSocket connection
        function initSocket() {
            socket = io();
            
            socket.on('connect', () => {
                console.log('Connected to server');
            });
            
            socket.on('progress_update', (data) => {
                updateProgress(data);
            });
            
            socket.on('qa_complete', (data) => {
                showResults(data);
            });
            
            socket.on('error', (data) => {
                alert('Error: ' + data.message);
            });
        }
        
        // Tab switching
        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.style.display = 'none';
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + 'Tab').style.display = 'block';
            
            // Add active class to clicked tab
            event.target.closest('.nav-tab').classList.add('active');
        }
        
        // File upload handling
        function handleFileUpload(files) {
            if (!files.length) return;
            
            const file = files[0];
            const formData = new FormData();
            formData.append('file', file);
            formData.append('data_type', document.getElementById('dataType').value);
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    currentSessionId = data.session_id;
                    showFilePreview(data.preview);
                    switchTab('qa');
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                alert('Upload failed: ' + error);
            });
        }
        
        // Show file preview
        function showFilePreview(preview) {
            const previewBody = document.getElementById('previewBody');
            previewBody.innerHTML = '';
            
            preview.forEach((row, index) => {
                if (index >= 5) return; // Show first 5 rows
                
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row.column}</td>
                    <td>${row.value}</td>
                    <td>${row.type}</td>
                `;
                previewBody.appendChild(tr);
            });
            
            document.getElementById('filePreview').style.display = 'block';
        }
        
        // Start QA process
        function startQA() {
            switchTab('qa');
            document.getElementById('progressSection').style.display = 'block';
        }
        
        // Update confidence value display
        function updateConfidenceValue(value) {
            document.getElementById('confidenceValue').textContent = value + '%';
        }
        
        // Run QA pipeline
        function runQAPipeline() {
            if (!currentSessionId) {
                alert('Please upload a file first');
                return;
            }
            
            const config = {
                confidence_threshold: parseInt(document.getElementById('confidenceSlider').value),
                auto_fix: document.getElementById('autoFix').checked,
                deduplication: document.getElementById('deduplication').checked,
                normalization: document.getElementById('normalization').checked
            };
            
            // Reset progress
            resetProgress();
            
            fetch('/run_qa', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    config: config
                })
            })
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                alert('Failed to start QA: ' + error);
            });
        }
        
        // Update progress visualization
        function updateProgress(data) {
            if (data.session_id !== currentSessionId) return;
            
            const stage = data.stage;
            const progress = data.progress;
            const message = data.message;
            
            // Update progress bar
            document.getElementById('progressFill').style.width = progress + '%';
            
            // Update stage circles
            const stages = ['loading', 'validation', 'normalization', 'deduplication', 'scoring', 'fixing', 'complete'];
            const stageIndex = stages.indexOf(stage);
            
            for (let i = 1; i <= 5; i++) {
                const circle = document.getElementById('stage' + i);
                if (i - 1 < stageIndex) {
                    circle.className = 'stage-circle completed';
                } else if (i - 1 === stageIndex) {
                    circle.className = 'stage-circle active';
                } else {
                    circle.className = 'stage-circle';
                }
            }
            
            // Update message
            document.getElementById('progressMessage').textContent = message;
            
            // If complete, switch to results
            if (stage === 'complete') {
                setTimeout(() => {
                    switchTab('results');
                }, 1000);
            }
        }
        
        // Reset progress
        function resetProgress() {
            document.getElementById('progressFill').style.width = '0%';
            for (let i = 1; i <= 5; i++) {
                document.getElementById('stage' + i).className = 'stage-circle';
            }
            document.getElementById('progressMessage').textContent = 'Starting QA pipeline...';
        }
        
        // Show QA results
        function showResults(data) {
            // Update metrics
            document.getElementById('qualityScore').textContent = data.metrics.quality_score;
            document.getElementById('totalRecords').textContent = data.metrics.total_records;
            document.getElementById('issuesFound').textContent = data.metrics.issues_found;
            document.getElementById('completeness').textContent = data.metrics.completeness + '%';
            document.getElementById('validity').textContent = data.metrics.validity + '%';
            
            // Update issues table
            const tableBody = document.getElementById('issuesTableBody');
            tableBody.innerHTML = '';
            
            data.issues.slice(0, 50).forEach(issue => {
                const tr = document.createElement('tr');
                let confidenceClass = 'issue-low';
                if (issue.confidence > 80) confidenceClass = 'issue-high';
                else if (issue.confidence > 60) confidenceClass = 'issue-medium';
                
                tr.innerHTML = `
                    <td>${issue.record_id}</td>
                    <td>${issue.field}</td>
                    <td>${issue.issue_type}</td>
                    <td class="${confidenceClass}">${issue.confidence}%</td>
                    <td>${issue.suggested_value}</td>
                `;
                tableBody.appendChild(tr);
            });
        }
        
        // Export data
        function exportData(format) {
            if (!currentSessionId) {
                alert('Please run QA analysis first');
                return;
            }
            
            window.location.href = `/export/${currentSessionId}/${format}`;
        }
        
        // Export report
        function exportReport() {
            if (!currentSessionId) {
                alert('Please run QA analysis first');
                return;
            }
            
            window.location.href = `/export_report/${currentSessionId}`;
        }
        
        // Show report preview
        function showReport() {
            if (!currentSessionId) {
                alert('Please run QA analysis first');
                return;
            }
            
            fetch(`/get_report/${currentSessionId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const reportContent = document.getElementById('reportContent');
                        reportContent.textContent = JSON.stringify(data.report, null, 2);
                        document.getElementById('reportPreview').style.display = 'block';
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('Failed to load report: ' + error);
                });
        }
        
        // Initialize on load
        window.onload = function() {
            initSocket();
        };
    </script>
</body>
</html>
"""

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main HTML page"""
    return HTML_TEMPLATE

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Validate file extension
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if file_ext not in Config.ALLOWED_EXTENSIONS:
            return jsonify({'success': False, 'error': f'File type not allowed. Use: {", ".join(Config.ALLOWED_EXTENSIONS)}'})
        
        # Generate session ID
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + str(hash(file.filename))[-8:]
        file_path = os.path.join(Config.UPLOAD_FOLDER, f'{session_id}.{file_ext}')
        
        # Save file
        file.save(file_path)
        
        # Get data type
        data_type_str = request.form.get('data_type', 'company')
        data_type = DataType.COMPANY if data_type_str == 'company' else DataType.PEOPLE
        
        # Create QA session
        qa_engine = B2BDataQACopilot()
        qa_engine.session_id = session_id
        
        if not qa_engine.load_data(file_path, data_type):
            return jsonify({'success': False, 'error': 'Failed to load file'})
        
        # Store session
        qa_sessions[session_id] = qa_engine
        file_sessions[session_id] = {
            'filename': filename,
            'filepath': file_path,
            'data_type': data_type.value,
            'upload_time': datetime.now().isoformat()
        }
        
        # Generate preview
        preview = []
        if qa_engine.data is not None and not qa_engine.data.empty:
            for col in qa_engine.data.columns[:10]:  # First 10 columns
                if col != 'record_id':
                    sample_value = str(qa_engine.data[col].iloc[0]) if len(qa_engine.data) > 0 else ''
                    preview.append({
                        'column': col,
                        'value': sample_value[:50] + ('...' if len(sample_value) > 50 else ''),
                        'type': str(qa_engine.data[col].dtype)
                    })
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'filename': filename,
            'records': len(qa_engine.data) if qa_engine.data is not None else 0,
            'preview': preview
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/run_qa', methods=['POST'])
def run_qa():
    """Run QA pipeline"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        config = data.get('config', {})
        
        if session_id not in qa_sessions:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        qa_engine = qa_sessions[session_id]
        
        # Update configuration
        qa_engine.confidence_threshold = config.get('confidence_threshold', 70)
        qa_engine.auto_fix_enabled = config.get('auto_fix', True)
        
        # Set up progress callback
        def progress_callback(progress_data):
            socketio.emit('progress_update', progress_data)
        
        qa_engine.set_progress_callback(progress_callback)
        
        # Run in background thread
        def run_qa_background():
            try:
                qa_engine.run_qa_pipeline()
                
                # Send completion
                socketio.emit('qa_complete', {
                    'session_id': session_id,
                    'metrics': qa_engine.metrics,
                    'issues': [issue.to_dict() for issue in qa_engine.issues[:100]],
                    'duplicate_clusters': [cluster.to_dict() for cluster in qa_engine.duplicate_clusters]
                })
            except Exception as e:
                socketio.emit('error', {
                    'session_id': session_id,
                    'message': f'QA pipeline error: {str(e)}'
                })
        
        # Start background thread
        thread = threading.Thread(target=run_qa_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'QA pipeline started'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_report/<session_id>')
def get_report(session_id):
    """Get QA report"""
    try:
        if session_id not in qa_sessions:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        qa_engine = qa_sessions[session_id]
        report = qa_engine.generate_qa_report()
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export/<session_id>/<format>')
def export_data(session_id, format):
    """Export cleaned data"""
    try:
        if session_id not in qa_sessions:
            return "Session not found", 404
        
        qa_engine = qa_sessions[session_id]
        
        if format not in ['csv', 'json']:
            return "Invalid format", 400
        
        data_str = qa_engine.get_cleaned_data(format)
        
        # Create response
        from io import BytesIO
        import mimetypes
        
        mimetype = 'text/csv' if format == 'csv' else 'application/json'
        filename = f'cleaned_data_{session_id}.{format}'
        
        return send_file(
            BytesIO(data_str.encode('utf-8')),
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return str(e), 500

@app.route('/export_report/<session_id>')
def export_report(session_id):
    """Export QA report as JSON"""
    try:
        if session_id not in qa_sessions:
            return "Session not found", 404
        
        qa_engine = qa_sessions[session_id]
        report = qa_engine.generate_qa_report()
        
        # Convert to pretty JSON
        report_json = json.dumps(report, indent=2, default=str)
        
        # Create response
        from io import BytesIO
        
        filename = f'qa_report_{session_id}.json'
        
        return send_file(
            BytesIO(report_json.encode('utf-8')),
            mimetype='application/json',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return str(e), 500

@app.route('/sample/<data_type>')
def create_sample_data(data_type):
    """Create sample data for testing"""
    try:
        import csv
        
        if data_type == 'company':
            data = [
                ['company_name', 'website', 'domain', 'country', 'industry', 'company_size', 'revenue', 'linkedin_url', 'founded_year'],
                ['TechCorp Inc', 'https://techcorp.com', 'techcorp.com', 'USA', 'Information Technology', '500', '100000000', 'https://linkedin.com/company/techcorp', '2010'],
                ['DataSystems', 'https://datasystems.io', 'datasystems.co', 'US', 'IT Services', '250', '50000000', '', '2015'],
                ['CloudSolutions', 'https://cloudsolutions.com', 'cloudsolutions.com', 'United States', 'Cloud Computing', '1000', '250000000', 'https://linkedin.com/company/cloudsolutions', '2008'],
                ['RetailTech', 'www.retailtech.com', 'retailtech.com', 'UK', 'Retail', '150', '30000000', 'https://linkedin.com/company/retailtech', '2012'],
                ['HealthInnovate', 'https://healthinnovate.com', 'healthinnovate.com', 'Canada', 'Healthcare', '300', '75000000', '', '2016']
            ]
            filename = 'sample_companies.csv'
        else:
            data = [
                ['full_name', 'job_title', 'email', 'phone', 'company_name', 'linkedin_profile'],
                ['John Smith', 'Senior Sales Manager', 'john.smith@techcorp.com', '+1-555-123-4567', 'TechCorp Inc', 'https://linkedin.com/in/johnsmith'],
                ['Jane Doe', 'Head of Marketing', 'jane.doe@techcorp.com', '5559876543', 'TechCorp Inc', 'https://linkedin.com/in/janedoe'],
                ['Bob Johnson', 'Software Engineer', 'bob.j@datasystems.io', '555-555-5555', 'DataSystems', ''],
                ['Alice Brown', 'CTO', 'alice@cloudsolutions.com', '4443332222', 'CloudSolutions', 'https://linkedin.com/in/alicebrown'],
                ['Charlie Wilson', 'Product Manager', 'charlie.wilson@retailtech.com', '+44-20-7946-0958', 'RetailTech', ''],
                ['David Lee', 'CEO', 'david.lee@healthinnovate.com', '4165551234', 'HealthInnovate', 'https://linkedin.com/in/davidlee']
            ]
            filename = 'sample_people.csv'
        
        # Save sample file
        sample_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        with open(sample_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': sample_path,
            'message': f'Sample {data_type} data created'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ============================================================================
# SOCKET.IO EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def run_cli():
    """Run the QA copilot from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="B2B Data QA Copilot - Command Line Interface")
    parser.add_argument("file", help="Path to CSV, JSON, or Excel file")
    parser.add_argument("--type", choices=["company", "people"], required=True, 
                       help="Type of data: company or people")
    parser.add_argument("--confidence", type=int, default=70, 
                       help="Confidence threshold (default: 70)")
    parser.add_argument("--output", choices=["csv", "json", "report"], default="report",
                       help="Output format (default: report)")
    parser.add_argument("--auto-fix", action="store_true", 
                       help="Apply automatic fixes")
    parser.add_argument("--sample", action="store_true",
                       help="Generate sample data instead")
    
    args = parser.parse_args()
    
    if args.sample:
        # Create sample data
        create_sample_data(args.type)
        print(f"Sample {args.type} data created in uploads folder")
        return
    
    # Initialize QA engine
    qa_engine = B2BDataQACopilot()
    qa_engine.confidence_threshold = args.confidence
    qa_engine.auto_fix_enabled = args.auto_fix
    
    # Load data
    data_type = DataType.COMPANY if args.type == "company" else DataType.PEOPLE
    if not qa_engine.load_data(args.file, data_type):
        print(f"Error: Failed to load {args.file}")
        return
    
    print(f"✓ Loaded {len(qa_engine.data)} records from {args.file}")
    print(f"⚙️ Running QA pipeline with confidence threshold: {args.confidence}%")
    
    # Run QA pipeline
    qa_engine.run_qa_pipeline()
    
    # Generate output
    if args.output == "csv":
        output = qa_engine.get_cleaned_data('csv')
        output_file = f"cleaned_{os.path.basename(args.file)}"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"✓ Cleaned data saved to: {output_file}")
    
    elif args.output == "json":
        output = qa_engine.get_cleaned_data('json')
        output_file = f"cleaned_{os.path.basename(args.file).split('.')[0]}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"✓ Cleaned data saved to: {output_file}")
    
    else:  # report
        report = qa_engine.generate_qa_report()
        
        print("\n" + "="*60)
        print("B2B DATA QA REPORT")
        print("="*60)
        print(f"\n📊 Data Type: {report['data_type']}")
        print(f"📈 Total Records: {report['total_records']}")
        print(f"🏆 Overall Quality Score: {report['quality_score']}/100")
        
        print("\n📋 Metrics:")
        for key, value in report['metrics'].items():
            if isinstance(value, float):
                print(f"  {key.replace('_', ' ').title()}: {value:.1f}%")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n⚠️ Issues Found: {len(qa_engine.issues)}")
        print("\n🔍 Issue Breakdown:")
        for issue_type, count in report['issues_by_type'].items():
            print(f"  {issue_type}: {count}")
        
        print(f"\n🔄 Duplicate Clusters: {len(report['duplicate_clusters'])}")
        
        # Save report to file
        report_file = f"qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Full report saved to: {report_file}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Run CLI mode
        run_cli()
    else:
        # Run web server
        print("\n" + "="*60)
        print("🧠 B2B Data QA Copilot - Offline Web Edition")
        print("="*60)
        print("\n🌐 Starting local web server...")
        print(f"📂 Upload folder: {os.path.abspath(Config.UPLOAD_FOLDER)}")
        print(f"📊 Results folder: {os.path.abspath(Config.RESULTS_FOLDER)}")
        print("\n🚀 Open your browser and navigate to:")
        print("   http://localhost:5000")
        print("\n📋 Available commands:")
        print("   python b2b_qa_app.py --help")
        print("   python b2b_qa_app.py sample.csv --type company")
        print("   python b2b_qa_app.py --sample --type company")
        print("\n⚡ Press Ctrl+C to stop the server")
        print("="*60 + "\n")
        
        # Start Flask app
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()