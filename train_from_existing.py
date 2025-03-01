#!/usr/bin/env python3
"""
Script to train the ML model from existing Paperless documents
This helps bootstrap the ML system with initial training data
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
from pymongo import MongoClient
import hashlib
import datetime
from typing import Dict, List, Any

# Load environment variables
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration
PAPERLESS_URL = os.getenv("PAPERLESS_URL", "http://your-paperless-instance:8000")
PAPERLESS_TOKEN = os.getenv("PAPERLESS_API_TOKEN")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "paperless_ml")
LIMIT = int(os.getenv("LIMIT", "100"))  # Number of documents to import

# Headers for Paperless API
HEADERS = {
    "Authorization": f"Token {PAPERLESS_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json; version=6",
}

# SSL Verification setting
DISABLE_SSL_VERIFY = os.getenv("DISABLE_SSL_VERIFY", "").lower() in ("true", "yes", "1")


def validate_env_vars():
    """Validate that all required environment variables are set"""
    missing_vars = []

    # Check required environment variables
    if not os.getenv("PAPERLESS_URL"):
        missing_vars.append("PAPERLESS_URL")
    if not os.getenv("PAPERLESS_API_TOKEN"):
        missing_vars.append("PAPERLESS_API_TOKEN")
    if not os.getenv("MONGO_URI"):
        missing_vars.append("MONGO_URI")

    if missing_vars:
        print(
            f"Error: The following required environment variables are missing: {', '.join(missing_vars)}"
        )
        print("Please set them in your .env file or environment.")
        return False

    return True


def get_documents(limit: int = 100) -> List[Dict]:
    """
    Fetch documents from Paperless that have good metadata for training.
    We'll look for documents that have titles and dates.

    Args:
        limit: Maximum number of documents to fetch

    Returns:
        List of document dictionaries
    """
    print(f"Fetching up to {limit} documents from Paperless...")

    all_documents = []
    page = 1

    # Look for documents that have a non-empty title and created_date
    query_params = "page_size=20"

    while True:
        # Fetch documents with content, with pagination
        url = f"{PAPERLESS_URL}/api/documents/?{query_params}&page={page}"
        print(f"Fetching page {page}...")

        response = requests.get(
            url,
            headers=HEADERS,
            verify=not DISABLE_SSL_VERIFY,
        )

        if response.status_code != 200:
            print(f"Error fetching documents: {response.status_code}")
            print(response.text)
            return []

        data = response.json()
        results = data.get("results", [])

        if not results:
            break

        # Filter for documents with good metadata
        good_docs = [
            doc
            for doc in results
            if doc.get("title")
            and doc.get("title") != "Untitled"
            and doc.get("created_date")
        ]

        all_documents.extend(good_docs)
        print(f"Found {len(good_docs)} documents with good metadata on page {page}")

        # Check if we've reached the last page or our limit
        if not data.get("next") or len(all_documents) >= limit:
            break

        page += 1

    # Apply the limit
    return all_documents[:limit]


def main():
    """Main function to import training data"""
    print("Starting Paperless ML training data import...")

    # Validate environment variables
    if not validate_env_vars():
        sys.exit(1)

    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    training_data = db.training_data

    # Get documents from Paperless
    documents = get_documents(limit=LIMIT)
    if not documents:
        print("No documents found for training. Exiting.")
        sys.exit(1)

    print(f"Found {len(documents)} documents with good metadata for training")

    # Import each document into the training database
    imported_count = 0
    skipped_count = 0

    for doc in documents:
        doc_id = doc["id"]
        title = doc["title"]
        created_date = doc.get("created_date", "")
        content = doc["content"]

        # Create content hash
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Check if this document already exists in the training data
        existing = training_data.find_one({"doc_id": doc_id})

        if existing:
            print(f"Document {doc_id} already exists in training data. Skipping.")
            skipped_count += 1
            continue

        # Insert document into training data
        training_data.insert_one(
            {
                "doc_id": doc_id,
                "content": content,
                "content_hash": content_hash,
                "title": title,
                "created_date": created_date,
                "created_at": datetime.datetime.now(),
                "updated_at": datetime.datetime.now(),
            }
        )

        imported_count += 1
        print(f"Imported document {doc_id}: '{title}' with date {created_date}")

    print(
        f"\nImport complete: {imported_count} documents imported, {skipped_count} skipped"
    )
    print("You can now run the main script to train the ML model with this data.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"ERROR: Script failed with exception: {e}")
        print("Detailed traceback:")
        traceback.print_exc()
        print("\nPlease check your configuration and try again.")
