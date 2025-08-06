#!/usr/bin/env python3
"""
Debug script to fix email address storage and test email functionality
"""

import json
from app import app, db
from models import Email
import logging

def fix_email_addresses():
    """Fix email addresses in database that are double-encoded JSON"""
    with app.app_context():
        emails = Email.query.all()
        
        for email in emails:
            changed = False
            
            # Fix to_addresses
            if email.to_addresses and isinstance(email.to_addresses, str):
                try:
                    # Try to parse as JSON string
                    parsed = json.loads(email.to_addresses)
                    email.to_addresses = parsed
                    changed = True
                    print(f"Fixed to_addresses for email {email.id}: {parsed}")
                except json.JSONDecodeError:
                    print(f"Could not parse to_addresses for email {email.id}: {email.to_addresses}")
            
            # Fix cc_addresses  
            if email.cc_addresses and isinstance(email.cc_addresses, str):
                try:
                    parsed = json.loads(email.cc_addresses)
                    email.cc_addresses = parsed
                    changed = True
                    print(f"Fixed cc_addresses for email {email.id}: {parsed}")
                except json.JSONDecodeError:
                    print(f"Could not parse cc_addresses for email {email.id}: {email.cc_addresses}")
            
            # Fix bcc_addresses
            if email.bcc_addresses and isinstance(email.bcc_addresses, str):
                try:
                    parsed = json.loads(email.bcc_addresses)
                    email.bcc_addresses = parsed
                    changed = True
                    print(f"Fixed bcc_addresses for email {email.id}: {parsed}")
                except json.JSONDecodeError:
                    print(f"Could not parse bcc_addresses for email {email.id}: {email.bcc_addresses}")
            
            if changed:
                db.session.commit()
                print(f"Updated email {email.id}")

def test_address_parsing():
    """Test the address parsing function"""
    def parse_addresses(addr_field):
        if not addr_field:
            return []
        # If it's already a list (new JSON format), return as is
        if isinstance(addr_field, list):
            return addr_field
        # If it's a string (old JSON string format), parse it
        if isinstance(addr_field, str):
            try:
                parsed = json.loads(addr_field)
                # Handle double-encoded JSON strings recursively
                if isinstance(parsed, str):
                    try:
                        parsed = json.loads(parsed)
                    except (json.JSONDecodeError, TypeError):
                        pass
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                return []
        return []
    
    # Test cases
    test_cases = [
        None,
        [],
        ["test@example.com"],
        '["test@example.com"]',
        '"[\\"test@example.com\\"]"',  # Double-encoded
        "invalid json",
        ""
    ]
    
    for test_case in test_cases:
        result = parse_addresses(test_case)
        print(f"Input: {test_case} -> Output: {result} (type: {type(result)})")

if __name__ == "__main__":
    print("=== Testing address parsing ===")
    test_address_parsing()
    
    print("\n=== Fixing database ===")
    fix_email_addresses()
    
    print("\n=== Done ===")