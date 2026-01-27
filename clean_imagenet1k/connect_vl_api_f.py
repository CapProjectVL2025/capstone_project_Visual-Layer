#!/usr/bin/env python3
"""
Visual Layer API Client - Fixed Version
========================================

This version includes enhanced error handling for export rejections.
"""

import os
import time
import requests
import jwt
from typing import Dict, Optional, Any
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

class VisualLayerAPIClient:
    """Client for Visual Layer Cloud API with JWT authentication."""
    
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 base_url: str = None, token_expiry_minutes: int = 60):
        self.base_url = base_url or os.environ.get(
            'VL_BASE_URL', 
            'https://app.visual-layer.com'
        )
        self.token_expiry_minutes = token_expiry_minutes
        
        # Load credentials from environment
        self.api_key = api_key or os.environ.get('api_key')
        self.api_secret = api_secret or os.environ.get('api_secret')
        
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "API credentials required. Set api_key and api_secret environment variables.\n"
                "Get credentials: https://app.visual-layer.com/api/v1/api_credentials"
            )
        
        # Token state
        self.token = None
        self.token_expires_at = None
        
        # Generate initial token
        self._generate_jwt_token()
    
    def _generate_jwt_token(self):
        """Generate JWT token using PyJWT library."""
        print("🔐 Generating JWT token...")
        
        try:
            jwt_algorithm = "HS256"
            jwt_header = {
                'alg': jwt_algorithm,
                'typ': 'JWT',
                'kid': self.api_key,
            }
            
            now = datetime.now(tz=timezone.utc)
            expiration = now + timedelta(minutes=self.token_expiry_minutes)
            
            payload = {
                'sub': self.api_key,
                'iat': int(now.timestamp()),
                'exp': int(expiration.timestamp()),
                'iss': 'sdk'
            }
            
            self.token = jwt.encode(
                payload=payload,
                key=self.api_secret,
                algorithm=jwt_algorithm,
                headers=jwt_header
            )
            
            self.token_expires_at = expiration - timedelta(minutes=5)
            print(f"  ✓ Token generated (expires in {self.token_expiry_minutes} min)")
            
        except Exception as e:
            raise Exception(
                f"JWT generation failed: {str(e)}\n"
                "Install PyJWT: pip install PyJWT"
            )
    
    def _ensure_valid_token(self):
        """Ensure token is valid, refresh if expired."""
        if not self.token or datetime.now(tz=timezone.utc) >= self.token_expires_at:
            print("🔄 Token expired, refreshing...")
            self._generate_jwt_token()
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with valid JWT bearer token."""
        self._ensure_valid_token()
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def test_connection(self, dataset_id: str) -> bool:
        """Test API connection by fetching dataset info."""
        url = f"{self.base_url}/api/v1/dataset/{dataset_id}"
        
        try:
            response = requests.get(url, headers=self.get_headers(), timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Connected to Visual Layer")
                print(f"  Dataset: {data.get('display_name', 'Unknown')}")
                print(f"  Status: {data.get('status', 'Unknown')}")
                print(f"  Progress: {data.get('progress', 0)}%")
                return True
            elif response.status_code == 401:
                print("✗ Authentication failed - check credentials")
                return False
            elif response.status_code == 404:
                print("✗ Dataset not found - check dataset ID")
                return False
            else:
                print(f"✗ Error {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Connection error: {str(e)}")
            return False
    
    def export_dataset(self, dataset_id: str, format: str = 'json',
                      include_images: bool = False, 
                      file_name: str = 'export.zip') -> str:
        """
        Start dataset export from Visual Layer.
        
        **FIXED**: Better error handling for API rejections.
        """
        print(f"📤 Starting dataset export...")
        print(f"  Format: {format}")
        print(f"  Include images: {include_images}")
        print(f"  File name: {file_name}")
        
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/dataset/{dataset_id}/export_context_async",
                headers={
                    **self.get_headers(),
                    "Accept": "application/json, text/plain, */*"
                },
                params={
                    "file_name": file_name,
                    "export_format": format,
                    "include_images": str(include_images).lower()
                },
                timeout=30
            )
            
            # Check HTTP status
            if response.status_code != 200:
                print(f"\n❌ Export request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()
            
            # Parse response
            try:
                data = response.json()
            except Exception as e:
                print(f"\n❌ Failed to parse response as JSON")
                print(f"Raw response: {response.text[:500]}")
                raise Exception(f"Invalid JSON response: {e}")
            
            # Check for error in response
            if 'error' in data:
                error_msg = data.get('error', 'Unknown error')
                print(f"\n❌ Export rejected by API")
                print(f"Error: {error_msg}")
                raise Exception(f"Export rejected: {error_msg}")
            
            # Check if we got a task ID
            if 'id' not in data:
                print(f"\n❌ No task ID in response")
                print(f"Response: {data}")
                raise Exception(f"No task ID returned. Response: {data}")
            
            task_id = data['id']
            print(f"  ✓ Export started (Task ID: {task_id})")
            return task_id
            
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Network error during export request")
            print(f"Error: {str(e)}")
            raise
        except Exception as e:
            print(f"\n❌ Unexpected error during export")
            print(f"Error: {str(e)}")
            raise
    
    def get_export_status(self, dataset_id: str, export_task_id: str) -> Dict[str, Any]:
        """Check export task status."""
        response = requests.get(
            f"{self.base_url}/api/v1/dataset/{dataset_id}/export_status",
            headers={
                **self.get_headers(),
                "Accept": "application/json, text/plain, */*"
            },
            params={"export_task_id": export_task_id},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_export(self, dataset_id: str, export_task_id: str,
                       poll_interval: int = 30, max_wait: int = 3600) -> str:
        """
        Poll export status until complete.
        
        **FIXED**: Handles multiple possible field names for download URL.
        """
        print("⏳ Waiting for export to complete...")
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < max_wait:
            try:
                status_data = self.get_export_status(dataset_id, export_task_id)
                status = status_data.get('status', 'PENDING')
                
                # Only print if status changed
                if status != last_status:
                    print(f"  📊 Status: {status}")
                    last_status = status
                
                if status == 'COMPLETED':
                    print("  ✓ Export complete")
                    
                    # Try multiple possible field names for download URL
                    url_fields = ['download_uri', 'download_url', 'downloadUri', 
                                'url', 'uri', 'file_url', 'downloadUrl']
                    
                    for field in url_fields:
                        if field in status_data and status_data[field]:
                            print(f"  ✓ Found download URL at field: {field}")
                            return status_data[field]
                    
                    # Check nested result
                    if 'result' in status_data and isinstance(status_data['result'], dict):
                        for field in url_fields:
                            if field in status_data['result'] and status_data['result'][field]:
                                print(f"  ✓ Found download URL in result.{field}")
                                return status_data['result'][field]
                    
                    # If still not found, print debug info
                    print(f"\n❌ Export completed but no download URL found")
                    print(f"Available fields: {list(status_data.keys())}")
                    print(f"Full response: {status_data}")
                    raise Exception(f"Export completed but no download URL found. Response: {status_data}")
                
                elif status == 'FAILED':
                    error = status_data.get('error', status_data.get('message', 'Unknown error'))
                    print(f"\n❌ Export failed")
                    print(f"Error: {error}")
                    print(f"Full response: {status_data}")
                    raise Exception(f"Export failed: {error}")
                
                elif status in ['PENDING', 'IN_PROGRESS', 'PROCESSING']:
                    # These are normal, keep waiting
                    elapsed = int(time.time() - start_time)
                    if elapsed % 60 == 0:  # Print every minute
                        print(f"  ⏳ Still {status.lower()}... (elapsed: {elapsed//60}m)")
                else:
                    # Unknown status
                    print(f"  ⚠️  Unknown status: {status}")
                    print(f"     Full response: {status_data}")
                
                time.sleep(poll_interval)
                
            except requests.exceptions.RequestException as e:
                print(f"  ⚠️  Network error checking status: {e}")
                print(f"     Will retry in {poll_interval}s...")
                time.sleep(poll_interval)
        
        raise TimeoutError(
            f"Export did not complete within {max_wait} seconds (last status: {last_status})"
        )
    
    def download_export(self, download_url: str, output_path: str):
        """Download export ZIP file."""
        print(f"⬇️  Downloading export to {output_path}...")
        
        response = requests.get(download_url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                f.write(chunk)
                
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"  ⬇️  {percent:.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end='\r')
        
        print(f"\n  ✓ Download complete ({downloaded / (1024**2):.1f} MB)")