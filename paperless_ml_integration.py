import requests
import json
import datetime
import re
import os
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
import urllib3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib
from pymongo import MongoClient
from dateutil import parser as date_parser

# Load environment variables from .env file
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration
PAPERLESS_URL = os.getenv("PAPERLESS_URL", "http://your-paperless-instance:8000")
PAPERLESS_TOKEN = os.getenv("PAPERLESS_API_TOKEN")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
MONGO_DB = os.getenv("MONGO_DB", "paperless_ml")
AUTO_PROCESSING = os.getenv("AUTO_PROCESSING", "False").lower() in ("true", "yes", "1")
MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "20"))
RECIPIENT_TAGS = os.getenv("RECIPIENT_TAGS", "").split(",")
RECIPIENT_TAGS = [tag.strip() for tag in RECIPIENT_TAGS if tag.strip()]

# Headers for Paperless API
HEADERS = {
    "Authorization": f"Token {PAPERLESS_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json; version=6",  # Using the latest API version
}

# SSL Verification setting
DISABLE_SSL_VERIFY = os.getenv("DISABLE_SSL_VERIFY", "").lower() in ("true", "yes", "1")
if DISABLE_SSL_VERIFY:
    # Disable SSL warnings if verification is disabled
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    print(
        "WARNING: SSL certificate verification is disabled. Use only with trusted servers."
    )

# Model paths
MODEL_DIR = Path("./models")
DATE_MODEL_PATH = MODEL_DIR / "date_model.joblib"
TITLE_MODEL_PATH = MODEL_DIR / "title_model.joblib"
TAGS_MODEL_PATH = MODEL_DIR / "tags_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"
DATE_VECTORIZER_PATH = MODEL_DIR / "date_vectorizer.joblib"
TAGS_VECTORIZER_PATH = MODEL_DIR / "tags_vectorizer.joblib"

# Create model directory if it doesn't exist
MODEL_DIR.mkdir(exist_ok=True)


class PaperlessML:
    def __init__(self):
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[MONGO_DB]
        self.training_data = self.db.training_data

        # Initialize tag ID mapping
        self.tag_ids = {}

        # Initialize or load models
        self.initialize_models()

    def initialize_models(self):
        """Initialize models or load existing ones"""
        # Title prediction model
        if TITLE_MODEL_PATH.exists() and VECTORIZER_PATH.exists():
            print("Loading existing title model and vectorizer...")
            self.title_model = joblib.load(TITLE_MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.has_title_model = True
        else:
            print("No existing title model found. Will create after gathering data.")
            self.title_model = None
            self.vectorizer = TfidfVectorizer(
                max_features=10000, stop_words="english", ngram_range=(1, 2)
            )
            self.has_title_model = False

        # Date prediction model
        if DATE_MODEL_PATH.exists() and DATE_VECTORIZER_PATH.exists():
            print("Loading existing date model and vectorizer...")
            self.date_model = joblib.load(DATE_MODEL_PATH)
            self.date_vectorizer = joblib.load(DATE_VECTORIZER_PATH)
            self.has_date_model = True
        else:
            print("No existing date model found. Will create after gathering data.")
            self.date_model = None
            self.date_vectorizer = TfidfVectorizer(
                max_features=5000, stop_words="english", ngram_range=(1, 2)
            )
            self.has_date_model = False

        # Tags prediction model
        if TAGS_MODEL_PATH.exists() and TAGS_VECTORIZER_PATH.exists():
            print("Loading existing tags model and vectorizer...")
            self.tags_model = joblib.load(TAGS_MODEL_PATH)
            self.tags_vectorizer = joblib.load(TAGS_VECTORIZER_PATH)
            self.has_tags_model = True
        else:
            print("No existing tags model found. Will create after gathering data.")
            self.tags_model = None
            self.tags_vectorizer = TfidfVectorizer(
                max_features=5000, stop_words="english", ngram_range=(1, 2)
            )
            self.has_tags_model = False

    def validate_env_vars(self):
        """Validate that all required environment variables are set"""
        missing_vars = []

        # Get all required environment variables
        paperless_url = os.getenv("PAPERLESS_URL", "")
        paperless_token = os.getenv("PAPERLESS_API_TOKEN", "")
        mongo_uri = os.getenv("MONGO_URI", "")

        if not paperless_url:
            missing_vars.append("PAPERLESS_URL")
        if not paperless_token:
            missing_vars.append("PAPERLESS_API_TOKEN")
        if not mongo_uri:
            missing_vars.append("MONGO_URI")

        if missing_vars:
            print(
                f"Error: The following required environment variables are missing: {', '.join(missing_vars)}"
            )
            print("Please set them in your .env file or environment.")
            return False

        return True

    def get_documents(
        self, processed_tag_id: Optional[int] = None, limit: int = 20
    ) -> List[Dict]:
        """
        Fetch documents from Paperless that haven't been processed yet.

        Args:
            processed_tag_id: ID of a tag used to mark documents as processed
            limit: Maximum number of documents to fetch

        Returns:
            List of document dictionaries
        """
        # If we have a processed tag, exclude documents with this tag
        tag_filter = f"&tags__id__none={processed_tag_id}" if processed_tag_id else ""

        all_documents = []
        page = 1

        while True:
            # Fetch documents with content, with pagination
            url = f"{PAPERLESS_URL}/api/documents/?page={page}&page_size=20{tag_filter}"
            print(f"Fetching: {url}")

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

            all_documents.extend(results)

            # Check if we've reached the last page or our limit
            if not data.get("next") or len(all_documents) >= limit:
                break

            page += 1

        # Apply the limit
        return all_documents[:limit]

    def get_or_create_tag(self, tag_name: str) -> Optional[int]:
        """Create or find a tag in Paperless by name"""
        # Check cache first
        if tag_name in self.tag_ids:
            return self.tag_ids[tag_name]

        # First try to find the tag
        from urllib.parse import quote

        encoded_name = quote(tag_name)

        response = requests.get(
            f"{PAPERLESS_URL}/api/tags/?name__icontains={encoded_name}",
            headers=HEADERS,
            verify=not DISABLE_SSL_VERIFY,
        )

        if response.status_code == 200 and response.json()["count"] > 0:
            # Found the tag, return its ID
            tag_id = response.json()["results"][0]["id"]
            self.tag_ids[tag_name] = tag_id
            return tag_id

        # Tag doesn't exist, create it
        print(f"Creating new tag '{tag_name}'...")

        body = {
            "name": tag_name,
            "color": "#3498db",  # Blue color
            "is_inbox_tag": False,
            "matching_algorithm": 6,
            "match": "",
        }

        # Create the tag
        response = requests.post(
            f"{PAPERLESS_URL}/api/tags/",
            headers=HEADERS,
            json=body,
            verify=not DISABLE_SSL_VERIFY,
        )

        if response.status_code in (200, 201):
            tag_id = response.json()["id"]
            self.tag_ids[tag_name] = tag_id
            print(f"Successfully created tag '{tag_name}' with ID {tag_id}")
            return tag_id

        print(f"Error creating tag: {response.status_code}")
        print(response.text)
        return None

    def get_or_create_ai_processed_tag(self) -> Optional[int]:
        """Create or find the 'AI' tag in Paperless"""
        return self.get_or_create_tag("AI")

    def update_document(
        self,
        doc_id: int,
        title: str,
        created_date: str,
        recipient_tag_ids: List[int],
        ai_processed_tag_id: int,
    ) -> bool:
        """
        Update a document with the provided title, date, and tags.

        Args:
            doc_id: Document ID
            title: New title for the document
            created_date: New created date (YYYY-MM-DD format)
            recipient_tag_ids: List of recipient tag IDs to add
            ai_processed_tag_id: ID of tag marking document as processed

        Returns:
            True if successful, False otherwise
        """
        # Get current document to get existing tags
        response = requests.get(
            f"{PAPERLESS_URL}/api/documents/{doc_id}/",
            headers=HEADERS,
            verify=not DISABLE_SSL_VERIFY,
        )

        if response.status_code != 200:
            print(f"Error getting document {doc_id}: {response.status_code}")
            return False

        doc_data = response.json()
        current_tags = doc_data.get("tags", [])

        # Add AI tag if not already present
        if ai_processed_tag_id not in current_tags:
            current_tags.append(ai_processed_tag_id)

        # Add recipient tags
        for tag_id in recipient_tag_ids:
            if tag_id not in current_tags:
                current_tags.append(tag_id)

        # Prepare update data
        update_data = {"title": title, "tags": current_tags}

        # Add created_date if it's valid
        if created_date:
            try:
                datetime.datetime.strptime(created_date, "%Y-%m-%d")
                update_data["created_date"] = created_date
            except ValueError:
                print(f"Invalid date format: {created_date}, skipping this field")

        # Update the document
        response = requests.patch(
            f"{PAPERLESS_URL}/api/documents/{doc_id}/",
            headers=HEADERS,
            json=update_data,
            verify=not DISABLE_SSL_VERIFY,
        )

        if response.status_code in (200, 201, 204):
            print(f"Successfully updated document {doc_id}")
            return True

        print(f"Error updating document {doc_id}: {response.status_code}")
        print(response.text)
        return False

    def predict_title(self, content: str) -> str:
        """
        Predict a title for a document based on its content.

        Args:
            content: Document content text

        Returns:
            Predicted title string
        """
        if not self.has_title_model:
            # No model exists yet, return empty string
            return ""

        # Vectorize the content
        content_vec = self.vectorizer.transform([content])

        # Make prediction
        title_words = self.title_model.predict(content_vec)

        # Return the predicted title
        if isinstance(title_words[0], str):
            return title_words[0]
        else:
            return "Untitled Document"  # Fallback

    def predict_date(self, content: str) -> str:
        """
        Predict a document date based on content.

        Args:
            content: Document content text

        Returns:
            Predicted date in YYYY-MM-DD format or empty string
        """
        if not self.has_date_model:
            # Use regex to find potential dates in the content
            # This is a basic fallback when no model exists
            return self.extract_date_from_content(content)

        # Use the model for prediction
        content_vec = self.date_vectorizer.transform([content])
        predicted_date = self.date_model.predict(content_vec)

        if isinstance(predicted_date[0], str):
            return predicted_date[0]
        else:
            return ""  # Fallback

    def predict_recipient_tags(self, content: str) -> List[str]:
        """
        Predict recipient tags for a document based on content.

        Args:
            content: Document content text

        Returns:
            List of predicted recipient tags
        """
        if not self.has_tags_model or not RECIPIENT_TAGS:
            return []

        # Use the model for prediction
        content_vec = self.tags_vectorizer.transform([content])
        predicted_tags = self.tags_model.predict(content_vec)

        # Convert binary predictions to tag names
        predicted_tag_names = []
        for i, is_present in enumerate(predicted_tags[0]):
            if is_present:
                predicted_tag_names.append(RECIPIENT_TAGS[i])

        return predicted_tag_names

    def extract_date_from_content(self, content: str) -> str:
        """
        Try to extract a date from document content using regex patterns.
        This is used as a fallback or to improve predictions.

        Args:
            content: Document content text

        Returns:
            Extracted date in YYYY-MM-DD format or empty string
        """
        # Define regex patterns for common date formats
        date_patterns = [
            r"\b\d{2}[./-]\d{2}[./-]\d{4}\b",  # DD.MM.YYYY, DD-MM-YYYY, DD/MM/YYYY
            r"\b\d{4}[./-]\d{2}[./-]\d{2}\b",  # YYYY.MM.DD, YYYY-MM-DD, YYYY/MM/DD
            r"\b\d{1,2}\.\s*[A-Za-z]+\s*\d{4}\b",  # DD. Month YYYY
            r"\b[A-Za-z]+\s*\d{1,2}[,.]?\s*\d{4}\b",  # Month DD, YYYY
        ]

        potential_dates = []

        # Extract all potential dates from the content
        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            potential_dates.extend(matches)

        if not potential_dates:
            return ""

        # Try to parse each potential date and return the first valid one
        for date_str in potential_dates:
            try:
                date_obj = date_parser.parse(date_str, fuzzy=True)
                # Return in YYYY-MM-DD format
                return date_obj.strftime("%Y-%m-%d")
            except:
                continue

        return ""  # Could not parse any dates

    def store_training_data(
        self,
        doc_id: int,
        content: str,
        title: str,
        created_date: str,
        recipient_tags: List[str],
    ):
        """
        Store document data for training.

        Args:
            doc_id: Document ID
            content: Document content text
            title: Document title
            created_date: Document created date (YYYY-MM-DD)
            recipient_tags: List of recipient tag names
        """
        # Sanitize all strings to remove invalid Unicode characters
        if content:
            # First attempt to replace invalid characters
            content = content.encode("utf-8", errors="replace").decode("utf-8")

            # Additionally remove surrogate characters with regex
            import re

            content = re.sub(r"[\ud800-\udfff]", "", content)

        # Also sanitize the title and recipient tags
        title = title.encode("utf-8", errors="replace").decode("utf-8")
        recipient_tags = [
            tag.encode("utf-8", errors="replace").decode("utf-8")
            for tag in recipient_tags
        ]

        # Create a document hash to avoid storing duplicate content
        content_hash = hashlib.md5(content.encode()).hexdigest()

        try:
            # Check if this document already exists in the training data
            existing = self.training_data.find_one({"doc_id": doc_id})

            if existing:
                # Update existing document
                self.training_data.update_one(
                    {"doc_id": doc_id},
                    {
                        "$set": {
                            "content": content,
                            "content_hash": content_hash,
                            "title": title,
                            "created_date": created_date,
                            "recipient_tags": recipient_tags,
                            "updated_at": datetime.datetime.now(),
                        }
                    },
                )
                print(f"Updated training data for document {doc_id}")
            else:
                # Insert new document
                self.training_data.insert_one(
                    {
                        "doc_id": doc_id,
                        "content": content,
                        "content_hash": content_hash,
                        "title": title,
                        "created_date": created_date,
                        "recipient_tags": recipient_tags,
                        "created_at": datetime.datetime.now(),
                        "updated_at": datetime.datetime.now(),
                    }
                )
                print(f"Stored new training data for document {doc_id}")

            # Train the models after each document update
            self.train_models()

        except Exception as e:
            print(f"Error storing training data: {e}")
            print("Continuing without storing this document's data.")
            # Don't re-raise the exception so processing can continue

    def train_models(self):
        """Train or retrain the ML models with collected data"""
        # Get all training data
        cursor = self.training_data.find()
        training_docs = list(cursor)

        if len(training_docs) < 2:
            print(
                "Not enough training data to build models (minimum 2 documents required)"
            )
            return False

        print(f"Training models with {len(training_docs)} documents")

        # Prepare data for title prediction
        contents = [doc["content"] for doc in training_docs]
        titles = [doc["title"] for doc in training_docs]

        # Create content vectors
        X_title = self.vectorizer.fit_transform(contents)

        # Train title model (treat as a classification problem for simplicity)
        self.title_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.title_model.fit(X_title, titles)

        # Save title model and vectorizer
        joblib.dump(self.title_model, TITLE_MODEL_PATH)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)

        # Set flag for title model
        self.has_title_model = True

        # Prepare data for date prediction
        # Filter out documents without a valid date
        date_training_docs = [doc for doc in training_docs if doc.get("created_date")]

        if len(date_training_docs) >= 2:
            date_contents = [doc["content"] for doc in date_training_docs]
            dates = [doc["created_date"] for doc in date_training_docs]

            # Create content vectors for date prediction
            X_date = self.date_vectorizer.fit_transform(date_contents)

            # Train date model (also as a classification problem)
            self.date_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.date_model.fit(X_date, dates)

            # Save date model and vectorizer
            joblib.dump(self.date_model, DATE_MODEL_PATH)
            joblib.dump(self.date_vectorizer, DATE_VECTORIZER_PATH)

            # Set flag for date model
            self.has_date_model = True
            print("Date model trained successfully")
        else:
            print(
                "Not enough documents with dates to train date model (minimum 2 required)"
            )

        # Prepare data for tags prediction if we have recipient tags defined
        if RECIPIENT_TAGS:
            # Create binary matrix for recipient tags
            tag_matrix = []
            for doc in training_docs:
                doc_tags = doc.get("recipient_tags", [])

                # Create binary vector for this document
                tag_vector = [1 if tag in doc_tags else 0 for tag in RECIPIENT_TAGS]
                tag_matrix.append(tag_vector)

            if len(tag_matrix) >= 2:
                # Create content vectors for tag prediction
                X_tags = self.tags_vectorizer.fit_transform(contents)

                # Train tags model (multi-label classification)
                self.tags_model = MultiOutputClassifier(
                    RandomForestClassifier(n_estimators=100, random_state=42)
                )
                self.tags_model.fit(X_tags, tag_matrix)

                # Save tags model and vectorizer
                joblib.dump(self.tags_model, TAGS_MODEL_PATH)
                joblib.dump(self.tags_vectorizer, TAGS_VECTORIZER_PATH)

                # Set flag for tags model
                self.has_tags_model = True
                print("Tags model trained successfully")
            else:
                print("Not enough documents with recipient tags to train tags model")

        print("Models trained and saved successfully")
        return True

    def clear_screen(self):
        """Clear the terminal screen"""
        # For Windows
        if os.name == "nt":
            os.system("cls")
        # For Mac and Linux
        else:
            os.system("clear")

    def get_next_unprocessed_document(
        self, processed_tag_id: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Fetch a single unprocessed document from Paperless.

        Args:
            processed_tag_id: ID of a tag used to mark documents as processed

        Returns:
            A single document dictionary or None if no unprocessed documents found
        """
        # If we have a processed tag, exclude documents with this tag
        tag_filter = f"&tags__id__none={processed_tag_id}" if processed_tag_id else ""

        # Fetch a single document with content
        url = f"{PAPERLESS_URL}/api/documents/?page=1&page_size=1{tag_filter}"
        print(f"Fetching: {url}")

        response = requests.get(
            url,
            headers=HEADERS,
            verify=not DISABLE_SSL_VERIFY,
        )

        if response.status_code != 200:
            print(f"Error fetching documents: {response.status_code}")
            print(response.text)
            return None

        data = response.json()
        results = data.get("results", [])

        if not results:
            return None

        # Sanitize the document content right after fetching
        if "content" in results[0]:
            try:
                # First attempt to replace invalid characters
                results[0]["content"] = (
                    results[0]["content"]
                    .encode("utf-8", errors="replace")
                    .decode("utf-8")
                )

                # Additionally remove surrogate characters with regex
                import re

                results[0]["content"] = re.sub(
                    r"[\ud800-\udfff]", "", results[0]["content"]
                )
            except Exception as e:
                print(f"Warning: Error sanitizing document content: {e}")
                # If there's an error, replace the content with a placeholder
                results[0][
                    "content"
                ] = "Content could not be processed due to encoding issues"

        return results[0]

    def process_documents(self):
        """Main function to process documents one at a time"""
        print("Starting Paperless ML document processing...")
        print(f"Will process up to {MAX_DOCUMENTS} documents in this run")

        # Validate environment variables
        print("Validating environment variables...")
        if not self.validate_env_vars():
            print("Environment validation failed. Exiting.")
            return

        # Get or create a tag for marking processed documents
        ai_processed_tag_id = self.get_or_create_ai_processed_tag()
        if not ai_processed_tag_id:
            print("Could not create or find AI tag")
            return

        # Process documents one by one up to the run limit
        processed_count = 0

        for i in range(MAX_DOCUMENTS):
            # Clear screen for each new document
            self.clear_screen()

            print(f"\nProcessing document {i+1}/{MAX_DOCUMENTS}")

            # Get the next unprocessed document
            doc = self.get_next_unprocessed_document(
                processed_tag_id=ai_processed_tag_id
            )

            if not doc:
                print("No more unprocessed documents found")
                break

            # Add document URL to display
            doc_url = f"{PAPERLESS_URL}/documents/{doc['id']}/details"

            print(f"Document ID: {doc['id']} - {doc['title']}")
            print(f"Document URL: {doc_url}")

            # Predict title and date based on content
            predicted_title = self.predict_title(doc["content"])

            # If no title prediction from model, use the existing one
            if not predicted_title:
                predicted_title = doc["title"]

            # Get date prediction
            predicted_date = self.predict_date(doc["content"])

            # If no date prediction from model, try to extract from content
            if not predicted_date:
                predicted_date = self.extract_date_from_content(doc["content"])

            # Fallback to existing date if extraction fails
            if not predicted_date and doc.get("created_date"):
                predicted_date = doc["created_date"]

            # Get recipient tag predictions
            predicted_tags = self.predict_recipient_tags(doc["content"])

            # Debug info to see what's happening with tag predictions
            print(f"DEBUG - Has tags model: {self.has_tags_model}")
            print(f"DEBUG - Predicted tags: {predicted_tags}")

            # In auto mode, use predictions without confirmation
            if AUTO_PROCESSING:
                print(f"Auto-applying title: '{predicted_title}'")
                print(f"Auto-applying date: '{predicted_date}'")
                print(f"Auto-applying recipient tags: {', '.join(predicted_tags)}")

                # Get tag IDs for recipient tags
                recipient_tag_ids = []
                for tag_name in predicted_tags:
                    tag_id = self.get_or_create_tag(tag_name)
                    if tag_id:
                        recipient_tag_ids.append(tag_id)

                # Store training data
                self.store_training_data(
                    doc["id"],
                    doc["content"],
                    predicted_title,
                    predicted_date,
                    predicted_tags,
                )

                # Update document in Paperless
                if self.update_document(
                    doc["id"],
                    predicted_title,
                    predicted_date,
                    recipient_tag_ids,
                    ai_processed_tag_id,
                ):
                    processed_count += 1

            else:
                # Manual mode - ask for confirmation
                print("\nDocument preview:")
                print(f"ID: {doc['id']}")
                print(f"URL: {doc_url}")
                print(f"Current title: {doc['title']}")
                print(f"Suggested title: {predicted_title}")

                try:
                    # Get user input for title with a timeout to avoid blocking
                    user_title = input(
                        f"Enter title (or press Enter to accept suggestion): "
                    ).strip()
                    final_title = user_title if user_title else predicted_title

                    # Display date info
                    print(f"Current date: {doc.get('created_date', 'None')}")
                    print(f"Suggested date: {predicted_date}")

                    # Get user input for date
                    user_date = input(
                        f"Enter date YYYY-MM-DD (or press Enter to accept suggestion): "
                    ).strip()
                    final_date = user_date if user_date else predicted_date

                    # Validate date format if provided
                    if final_date:
                        try:
                            datetime.datetime.strptime(final_date, "%Y-%m-%d")
                        except ValueError:
                            print(
                                f"Invalid date format: {final_date}, using empty date"
                            )
                            final_date = ""

                    # Display recipient tag options
                    if RECIPIENT_TAGS:
                        print("\nRecipient tags (comma-separated numbers or 'none'):")
                        for i, tag in enumerate(RECIPIENT_TAGS):
                            # Fix: Clearly mark if a tag is predicted
                            selected = "✓" if tag in predicted_tags else " "
                            print(f"  [{selected}] {i+1}. {tag}")

                        # Show which tags are suggested
                        if predicted_tags:
                            suggested_indices = [
                                RECIPIENT_TAGS.index(tag) + 1
                                for tag in predicted_tags
                                if tag in RECIPIENT_TAGS
                            ]
                            suggested_str = ",".join(
                                str(idx) for idx in suggested_indices
                            )
                            print(f"Suggested recipients: {suggested_str}")
                        else:
                            suggested_str = "none"
                            print("No recipients suggested by model")

                        tag_input = (
                            input(
                                f"Select recipient tags (or press Enter to accept suggestion): "
                            )
                            .strip()
                            .lower()
                        )

                        # Use the predicted tags as default if user just presses Enter
                        if not tag_input:
                            final_tags = predicted_tags
                        elif tag_input == "none":
                            final_tags = []
                        else:
                            try:
                                # Parse the comma-separated input
                                selected_indices = [
                                    int(i.strip()) - 1 for i in tag_input.split(",")
                                ]
                                final_tags = [
                                    RECIPIENT_TAGS[i]
                                    for i in selected_indices
                                    if 0 <= i < len(RECIPIENT_TAGS)
                                ]
                            except ValueError:
                                print("Invalid input, using suggested tags")
                                final_tags = predicted_tags
                    else:
                        final_tags = []

                    # Get tag IDs for recipient tags
                    recipient_tag_ids = []
                    for tag_name in final_tags:
                        tag_id = self.get_or_create_tag(tag_name)
                        if tag_id:
                            recipient_tag_ids.append(tag_id)

                    # Store training data with user-confirmed values
                    self.store_training_data(
                        doc["id"], doc["content"], final_title, final_date, final_tags
                    )

                    # Update document in Paperless
                    if self.update_document(
                        doc["id"],
                        final_title,
                        final_date,
                        recipient_tag_ids,
                        ai_processed_tag_id,
                    ):
                        processed_count += 1

                except EOFError:
                    print(
                        "ERROR: Input not available. Is this running in a non-interactive environment?"
                    )
                    print(
                        "Try running with docker-compose run --rm paperless-ml or set AUTO_PROCESSING=True"
                    )
                    return

        print(f"\nProcessed {processed_count} out of {MAX_DOCUMENTS} documents")

        print("\nDocument processing completed")


def main():
    """Main entry point"""
    try:
        paperless_ml = PaperlessML()
        paperless_ml.process_documents()
    except Exception as e:
        import traceback

        print(f"ERROR: Script failed with exception: {e}")
        print("Detailed traceback:")
        traceback.print_exc()
        print("\nPlease check your configuration and try again.")


if __name__ == "__main__":
    main()
