#!/usr/bin/env python3
"""
Visual Layer API Client - Research Grade
=========================================

Handles JWT authentication and API operations for Visual Layer Cloud.

Authentication Method:
    Uses PyJWT to generate JWT tokens from API credentials.
    Tokens expire after 60 minutes and are auto-refreshed.

Usage:
    # Set credentials in environment
    export VL_API_KEY="your-api-key"
    export VL_API_SECRET="your-api-secret"
    
    # Use in Python
    from vl_api_client import VisualLayerAPIClient
    client = VisualLayerAPIClient()
    
    # Test connection
    client.test_connection(dataset_id)
    
    # Export dataset
    task_id = client.export_dataset(dataset_id, include_images=True)
    download_url = client.wait_for_export(dataset_id, task_id)
    client.download_export(download_url, "export.zip")

Requirements:
    pip install PyJWT requests

Documentation:
    Visual Layer API: https://docs.visual-layer.com
    
Author: Research Team
Date: January 2026
License: MIT
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

dotenv_path = Path('/Users/saeedarellano/visual_layer/capstone_project_Visual-Layer/.env')
load_dotenv(dotenv_path=dotenv_path)


class VisualLayerAPIClient:
    """
    Client for Visual Layer Cloud API with JWT authentication.
    
    This client handles:
    - JWT token generation using API credentials
    - Automatic token refresh (tokens expire after 60 min)
    - Dataset export and download
    - Error handling and progress tracking
    
    Attributes:
        base_url (str): Visual Layer API base URL
        api_key (str): Visual Layer API key
        api_secret (str): Visual Layer API secret
        token (str): Current JWT token
        token_expires_at (datetime): Token expiration time
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 base_url: str = None, token_expiry_minutes: int = 60):
        """
        Initialize Visual Layer API client.
        
        Args:
            api_key: API key (or set VL_API_KEY env var)
            api_secret: API secret (or set VL_API_SECRET env var)
            base_url: API URL (default: https://app.visual-layer.com)
            token_expiry_minutes: Token validity duration (default: 60)
        
        Raises:
            ValueError: If API credentials are not provided
        
        Example:
            >>> client = VisualLayerAPIClient()
            >>> client.test_connection("dataset-id-123")
        """
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
                "API credentials required. Set VL_API_KEY and VL_API_SECRET.\n"
                "Get credentials: https://app.visual-layer.com/api/v1/api_credentials"
            )
        
        # Token state
        self.token = None
        self.token_expires_at = None
        
        # Generate initial token
        self._generate_jwt_token()
    
    def _generate_jwt_token(self):
        """
        Generate JWT token using PyJWT library.
        
        This implements Visual Layer's official authentication method:
        https://docs.visual-layer.com/authentication
        
        The JWT token includes:
        - API key in header (kid field)
        - Expiration timestamp
        - Issued-at timestamp
        - Subject (API key)
        """
        print("🔐 Generating JWT token...")
        
        try:
            # JWT configuration (Visual Layer standard)
            jwt_algorithm = "HS256"
            jwt_header = {
                'alg': jwt_algorithm,
                'typ': 'JWT',
                'kid': self.api_key,
            }
            
            # Token expiration
            now = datetime.now(tz=timezone.utc)
            expiration = now + timedelta(minutes=self.token_expiry_minutes)
            
            payload = {
                'sub': self.api_key,
                'iat': int(now.timestamp()),
                'exp': int(expiration.timestamp()),
                'iss': 'sdk'
            }
            
            # Generate JWT
            self.token = jwt.encode(
                payload=payload,
                key=self.api_secret,
                algorithm=jwt_algorithm,
                headers=jwt_header
            )
            
            # Track expiration (refresh 5 min early to be safe)
            self.token_expires_at = expiration - timedelta(minutes=5)
            
            print(f"  ✓ Token generated (expires in {self.token_expiry_minutes} min)")
            
        except Exception as e:
            raise Exception(
                f"JWT generation failed: {str(e)}\n"
                "Install PyJWT: pip install PyJWT"
            )
    
    def _ensure_valid_token(self):
        """
        Ensure token is valid, refresh if expired.
        
        Automatically called before each API request to prevent
        authentication failures during long-running operations.
        """
        if not self.token or datetime.now(tz=timezone.utc) >= self.token_expires_at:
            print("🔄 Token expired, refreshing...")
            self._generate_jwt_token()
    
    def get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers with valid JWT bearer token.
        
        Returns:
            Dict with Authorization header and Content-Type
        
        Example:
            >>> headers = client.get_headers()
            >>> response = requests.get(url, headers=headers)
        """
        self._ensure_valid_token()
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def test_connection(self, dataset_id: str) -> bool:
        """
        Test API connection by fetching dataset info.
        
        Args:
            dataset_id: Dataset ID to test with
            
        Returns:
            True if connection successful, False otherwise
        
        Example:
            >>> if client.test_connection("abc-123"):
            ...     print("Connected!")
        """
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
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get dataset metadata and status.
        
        Args:
            dataset_id: Visual Layer dataset ID
            
        Returns:
            Dictionary with dataset information
            
        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.base_url}/api/v1/dataset/{dataset_id}"
        response = requests.get(url, headers=self.get_headers(), timeout=30)
        response.raise_for_status()
        return response.json()
    
    def export_dataset(self, dataset_id: str, format: str = "json",
                      include_images: bool = False, 
                      file_name: str = "export.zip") -> str:
        """
        Start asynchronous dataset export.
        
        Uses Visual Layer's export_context_async endpoint to export
        dataset metadata and optionally images.
        
        Args:
            dataset_id: Visual Layer dataset ID
            format: Export format ('json' or 'parquet')
            include_images: Whether to include image files
            file_name: Name for output ZIP file
            
        Returns:
            Export task ID for polling status
            
        Example:
            >>> task_id = client.export_dataset("abc-123", include_images=True)
            >>> url = client.wait_for_export("abc-123", task_id)
            >>> client.download_export(url, "dataset.zip")
        """
        print(f"📤 Starting dataset export...")
        print(f"  Format: {format}")
        print(f"  Include images: {include_images}")
        
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
        response.raise_for_status()
        
        task_id = response.json()['id']
        print(f"  ✓ Export started (Task ID: {task_id})")
        return task_id
    
    def get_export_status(self, dataset_id: str, export_task_id: str) -> Dict[str, Any]:
        """
        Check export task status.
        
        Args:
            dataset_id: Visual Layer dataset ID
            export_task_id: Export task ID from export_dataset()
            
        Returns:
            Dictionary with status and download_uri (when ready)
        """
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
        
        Args:
            dataset_id: Visual Layer dataset ID
            export_task_id: Export task ID from export_dataset()
            poll_interval: Seconds between status checks (default: 30)
            max_wait: Maximum seconds to wait (default: 3600 = 1 hour)
            
        Returns:
            Download URL for completed export
            
        Raises:
            TimeoutError: If export doesn't complete within max_wait
            Exception: If export fails
        """
        print("⏳ Waiting for export to complete...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_data = self.get_export_status(dataset_id, export_task_id)
            print("DEBUG export_status:", status_data)
            status = status_data.get('status', 'PENDING')
            
            if status == 'COMPLETED':
                print("  ✓ Export complete")
                return status_data['download_uri']
            elif status == 'FAILED':
                error = status_data.get('error', 'Unknown error')
                raise Exception(f"Export failed: {error}")
            else:
                elapsed = int(time.time() - start_time)
                print(f"  ⏳ Status: {status} (elapsed: {elapsed}s)")
                time.sleep(poll_interval)
        
        raise TimeoutError(
            f"Export did not complete within {max_wait} seconds"
        )
    
    def download_export(self, download_url: str, output_path: str):
        """
        Download export ZIP file.
        
        Args:
            download_url: Download URL from wait_for_export()
            output_path: Local path to save ZIP file
            
        Example:
            >>> client.download_export(download_url, "imagenet.zip")
        """
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


def main():
    """
    Command-line interface for testing the API client.
    
    Usage:
        python vl_api_client.py --dataset-id YOUR_DATASET_ID
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visual Layer API Client - Test Connection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test connection
    python vl_api_client.py --dataset-id abc-123-def-456
    
    # Export dataset
    python vl_api_client.py --dataset-id abc-123 --export --output dataset.zip

Environment Variables:
    VL_API_KEY       Visual Layer API key (required)
    VL_API_SECRET    Visual Layer API secret (required)
    
Get Credentials:
    https://app.visual-layer.com/api/v1/api_credentials
        """
    )
    
    parser.add_argument('--dataset-id', required=True, 
                       help='Dataset ID to test')
    parser.add_argument('--export', action='store_true',
                       help='Export dataset after testing connection')
    parser.add_argument('--output', default='export.zip',
                       help='Output filename for export (default: export.zip)')
    parser.add_argument('--include-images', action='store_true',
                       help='Include images in export')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Visual Layer API Client - Connection Test")
    print("=" * 70)
    
    # Check environment
    api_key = os.environ.get('api_key', 'Not set')
    api_secret = os.environ.get('api_secret', 'Not set')
    
    print(f"API Key: {'Set' if len(api_key) > 20 else 'Not set'}")
    print(f"API Secret: {'Set' if len(api_secret) > 10 else 'Not set'}")
    print(f"Dataset ID: {args.dataset_id}")
    print("=" * 70 + "\n")
    
    try:
        # Initialize client
        client = VisualLayerAPIClient()
        
        # Test connection
        if not client.test_connection(args.dataset_id):
            print("\n❌ Connection test failed")
            print("\nTroubleshooting:")
            print("  1. Get credentials: https://app.visual-layer.com/api/v1/api_credentials")
            print("  2. Set environment:")
            print("     export VL_API_KEY='your-key'")
            print("     export VL_API_SECRET='your-secret'")
            print("  3. Verify dataset ID in Visual Layer UI")
            return 1
        
        # Export if requested
        if args.export:
            print("\n" + "=" * 70)
            print("Starting Export")
            print("=" * 70 + "\n")
            
            task_id = client.export_dataset(
                dataset_id=args.dataset_id,
                include_images=args.include_images
            )
            
            download_url = client.wait_for_export(args.dataset_id, task_id)
            client.download_export(download_url, args.output)
            
            print(f"\n✅ Export complete: {args.output}")
        
        print("\n" + "=" * 70)
        print("✅ All operations successful!")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())