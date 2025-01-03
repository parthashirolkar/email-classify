# Email Spam Classification System with Gmail Integration

This project implements an automated email spam classification system that integrates with Gmail using IMAP. It features a machine learning model for spam detection and a Python script for periodic email processing and classification.

The system automatically checks for new emails, classifies them as spam or ham (non-spam), and updates email labels accordingly. It is designed to run as a cronjob, periodically processing unread emails in the inbox.

## Repository Structure

- `email_triggers.py`: Main Python script for email processing and classification
- `model_training.py`: Script for training and saving the spam classification model

Key integration points:
- Gmail IMAP server connection
- Machine learning model integration (MultinomialNB classifier)

## Usage Instructions

### Installation

Prerequisites:
- Python 3.7+
- pip package manager

Steps:
1. Clone the repository
2. Install required packages:
   ```
   pip install python-dotenv pandas scikit-learn nltk matplotlib joblib pandarallel matplotlib
   ```
3. Download NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

### Configuration

1. Create a `.env` file in the project root with your Gmail credentials:
   ```
   EMAIL_ID=your.email@gmail.com
   PASSWORD=your_app_password
   ```
   Note: Use an app password for Gmail, not your regular account password.

2. Ensure you have a CSV file named `combined_data.csv` with columns 'text' and 'spam' for training the model.

### Using Pre-trained Model

If you don't want to train the model from scratch, you can use the pre-trained model:

1. Ensure the `model.pkl` and `vectorizer.pkl` files are present in the `obj/` directory.
2. If these files are not present, you'll need to train the model first.

Using the pre-trained model allows you to skip the model training step and start using the script immediately.

### Training the Model

If you choose to train the model yourself:

1. Prepare your training data and save it as `combined_data.csv` in the `csv/` directory. This file should contain 'text' and 'label' columns.

2. Run the model training script:
   ```
   python model_training.py
   ```
   This will create `model.pkl` and `vectorizer.pkl` in the `obj/` directory.

### Running the Script

To run the email classification script:

```
python email_triggers.py
```

This script is designed to be run periodically as a cronjob. You can set it up to run at regular intervals using your system's cron scheduler or a task scheduling service.

For example, to run the script every 15 minutes using cron, you could add the following line to your crontab:

```
*/15 * * * * /path/to/your/python /path/to/your/email_triggers.py
```

Make sure to replace `/path/to/your/python` with the actual path to your Python interpreter and `/path/to/your/email_triggers.py` with the full path to the script.

### Troubleshooting

1. IMAP Connection Issues:
   - Ensure your Gmail account has IMAP enabled in settings.
   - If using 2-factor authentication, generate an app password for this application.
   - Check your `.env` file for correct credentials.

2. Model Training Errors:
   - Verify `combined_data.csv` exists and has the correct format.
   - Ensure all required packages are installed.

3. Script Execution Errors:
   - Check the `email_classifier.log` file for detailed error messages.
   - Verify that `model.pkl` and `vectorizer.pkl` exist in the `obj/` directory.

### Performance Optimization

- Monitor the execution time of the script, especially for large inboxes.
- Consider implementing batch processing for large numbers of emails.
- Optimize the `clean_text` function if processing speed becomes a bottleneck.

## Data Flow

The email classification system follows this data flow:

1. The script is triggered periodically (e.g., by a cron job).
2. The script connects to Gmail via IMAP.
3. Unread emails are fetched from the inbox.
4. Each email's subject and body are extracted and preprocessed.
5. The preprocessed text is vectorized using the pre-trained CountVectorizer.
6. The vectorized text is passed through the Naive Bayes classifier.
7. Based on the classification result, the email is labeled as spam or ham in Gmail.
8. Classification results are logged.

```
[Cron Job] -> [Python Script] -> [Gmail IMAP] -> [Email Extraction]
                                                       |
                                                       v
[Classification Result] <- [Naive Bayes Model] <- [Text Vectorization]
        |
        v
[Gmail Label Update] -> [Logging]
```

Note: The model training process is separate and occurs offline before the script is deployed.