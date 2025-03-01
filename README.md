# Paperless ML Metadata Extractor

A machine learning tool that learns from your document processing decisions to improve title and date extraction for Paperless documents.

## Key Features

- **Self-improving ML model**: The system learns from your corrections to make better predictions over time
- **Focus on titles and dates**: Automatically predicts document titles and creation dates
- **Recipient tagging**: Learns to identify which person a document is relevant for
- **Revolving training approach**: After each run, the model is retrained with the new data
- **MongoDB integration**: Stores all training decisions for continuous improvement
- **Simple user interface**: Easily confirm or correct suggestions with a simple prompt
- **Docker-ready**: Runs in a containerized environment with MongoDB

## How It Works

1. The script pulls up to 20 documents without the "AI-Processed" tag from Paperless
2. For each document, it predicts the title and date based on previous decisions
3. You can accept the suggestions or provide corrections
4. All decisions are stored in MongoDB for model training
5. After processing, the ML model is retrained with all accumulated data
6. The next run will have improved predictions

## Setup with Docker

### Prerequisites

- Docker and Docker Compose
- Access to a Paperless instance with API token

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/paperless-ml-metadata.git
   cd paperless-ml-metadata
   ```

2. Create a `.env` file:
   ```
   cp .env.example .env
   ```

3. Edit the `.env` file with your credentials and configuration:
   ```
   PAPERLESS_URL=http://your-paperless-instance:8000
   PAPERLESS_API_TOKEN=your_paperless_api_token_here

   # Configure recipient tags - who are the documents for?
   RECIPIENT_TAGS=Greg,Carol,Marcia,Jan,Cindy,Greg,Peter,Bobby,Alice
   ```

4. Build and start the Docker containers:
   ```
   docker-compose up -d
   ```

### Running the Tool

To run the tool in interactive mode (for manual confirmation):

```bash
# Start MongoDB in the background
docker-compose up -d mongodb

# Run the paperless-ml container in interactive mode
docker-compose run --rm paperless-ml
```

To run in automatic mode (once you're confident in the model):

1. Edit the `.env` file:
   ```
   AUTO_PROCESSING=True
   ```

2. Run the container:
   ```
   docker-compose run --rm paperless-ml
   ```

## Setting Up Automated Document Processing

You can set up automatic document processing on a schedule using cron:

1. Make the `run_auto.sh` script executable:
   ```bash
   chmod +x run_auto.sh
   ```

2. Edit your crontab:
   ```bash
   crontab -e
   ```

3. Add one of the following lines to run the script on your preferred schedule:

   ```
   # Run every day at 2 AM
   0 2 * * * /path/to/paperless-ml-metadata/run_auto.sh >> /path/to/paperless-ml-metadata/cron.log 2>&1

   # Run every Sunday at 3 AM
   0 3 * * 0 /path/to/paperless-ml-metadata/run_auto.sh >> /path/to/paperless-ml-metadata/cron.log 2>&1

   # Run on the 1st and 15th of every month at 4 AM
   0 4 1,15 * * /path/to/paperless-ml-metadata/run_auto.sh >> /path/to/paperless-ml-metadata/cron.log 2>&1
   ```

### Cron Schedule Format

The cron format has five fields:
```
minute hour day month weekday command
```

- **minute**: 0-59
- **hour**: 0-23
- **day**: 1-31
- **month**: 1-12 (or names)
- **weekday**: 0-6 (Sunday is 0 or 7)

### Monitoring

The example above redirects both standard output and errors to a log file. You can check this file to monitor the script's execution:

```bash
tail -f /path/to/paperless-ml-metadata/cron.log
```

### Portainer Setup

If you're using Portainer with the recurring job feature:

1. Go to your Portainer instance
2. Navigate to "Stacks" > select your stack
3. Click on "Add a recurring job"
4. Set your schedule using cron syntax
5. Command: `/bin/bash -c "/path/to/paperless-ml-metadata/run_auto.sh"`
6. Select the host or node where the job should run
7. Save the job

## How to Improve Predictions

The ML models improve as they receive more training data. To get better predictions:

1. **Process more documents**: The more documents you process, the better the models become
2. **Provide corrections**: Don't just accept incorrect suggestions - provide the correct values
3. **Remove the AI-Processed tag**: If you want to reprocess a document, remove the tag in Paperless

## Auto Mode

Once you have sufficient training data and you're confident in the model's predictions, you can enable automatic processing:

1. Edit the `.env` file:
   ```
   AUTO_PROCESSING=True
   ```

2. Restart the container:
   ```
   docker-compose restart paperless-ml
   ```

## Docker Stack Setup in Portainer

If you're using Portainer, follow these steps:

1. Go to your Portainer instance
2. Navigate to Stacks > Add stack
3. Name your stack (e.g., "paperless-ml")
4. Upload the `docker-compose.yml` file or paste its contents
5. Click "Deploy the stack"
6. Check to ensure all services start properly
7. To run the tool manually, go to Containers, find the paperless-ml container, and click "Console" to run the script

## Technical Details

### Machine Learning Implementation

The system uses three separate machine learning models:

1. **Title Prediction Model**: A Random Forest Classifier trained on document content to predict appropriate titles
2. **Date Prediction Model**: A Random Forest Classifier that identifies creation dates based on document content
3. **Recipient Tag Model**: A Multi-Output Classifier that predicts which person(s) a document is relevant for

For documents where the ML model isn't confident or doesn't have enough training data, the system uses regex pattern matching as a fallback to extract dates from the document content.

### Recipient Tagging

The recipient tagging system helps categorize documents by the person they're relevant to:

1. **Configuration**: Define possible recipients in your `.env` file with the `RECIPIENT_TAGS` variable
   ```
   RECIPIENT_TAGS=Greg,Carol,Marcia,Jan,Cindy,Greg,Peter,Bobby,Alice
   ```

2. **Learning Process**: As you process documents, you'll be asked to identify which recipient(s) each document is for
   - The system will present a numbered list of possible recipients
   - You can select multiple recipients by entering comma-separated numbers
   - Or accept the system's suggestions by pressing Enter
   - Or specify "none" if the document isn't relevant to any recipient

3. **Prediction Improvement**: Over time, the system learns patterns in document content to predict the correct recipient(s)

4. **Tag Application**: Selected recipients will be added as tags to the document in Paperless, making them searchable and filterable

This feature is particularly useful for family or business documents where different documents may be relevant to different people.

### Data Storage

The MongoDB database stores:

- Document ID from Paperless
- Content hash (to prevent duplicate training)
- User-confirmed title
- User-confirmed date
- Timestamps for creation and updates

If you re-process a document by removing the AI-Processed tag in Paperless, the system will update the existing MongoDB record rather than creating a duplicate.

### Model Storage

The trained models are stored in the `./models` directory, which is persisted as a Docker volume:

- `title_model.joblib`: The Random Forest model for title predictions
- `vectorizer.joblib`: The TF-IDF vectorizer for converting document text to features
- `date_model.joblib`: The Random Forest model for date predictions
- `date_vectorizer.joblib`: The TF-IDF vectorizer for date extraction

### Customizing the Models

For advanced users who want to modify the machine learning models:

1. Stop the running containers:
   ```
   docker-compose down
   ```

2. Modify the `paperless_ml_integration.py` script
3. Rebuild the containers:
   ```
   docker-compose build
   ```

4. Restart the stack:
   ```
   docker-compose up -d
   ```

## Troubleshooting

### Common Issues

- **Connection Error**: Ensure your Paperless URL is accessible from the Docker container
- **Authentication Error**: Verify your API token is correct and has sufficient permissions
- **No Documents Found**: Check that documents exist without the AI-Processed tag
- **MongoDB Connection Failure**: Ensure the MongoDB container is running properly
- **Unicode Encoding Errors**: If you encounter encoding issues with document content, the script has been updated to handle these automatically by sanitizing invalid characters

### Resetting the System

If you need to start fresh with a new model:

1. Stop the containers:
   ```
   docker-compose down
   ```

2. Remove the models:
   ```
   rm -rf ./models/*
   ```

3. Restart the containers:
   ```
   docker-compose up -d
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.