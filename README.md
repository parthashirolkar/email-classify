# Email Spam Classification System with Gmail Integration

This project implements an automated email spam classification system that integrates with Gmail using IMAP. It features a machine learning model for spam detection and a FastAPI application for email processing and classification.

The system automatically checks for new emails, classifies them as spam or ham (non-spam), and updates email labels accordingly. It also provides an API endpoint for classifying individual emails based on their subject and body.

## Repository Structure

- `email_triggers.py`: Main FastAPI application for email processing and classification
- `model_training.py`: Script for training and saving the spam classification model
- `test.py`: Script for testing Gmail IMAP connection

Key integration points:
- Gmail IMAP server connection
- Machine learning model integration (MultinomialNB classifier)
- FastAPI endpoints for email classification and inbox processing

## Usage Instructions

### Installation

Prerequisites:
- Python 3.7+
- pip package manager

Steps:
1. Clone the repository
2. Install required packages:
   ```
   pip install fastapi uvicorn python-dotenv pandas scikit-learn nltk matplotlib joblib pandarallel
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

2. Ensure you have a CSV file named `emails.csv` with columns 'text' and 'spam' for training the model.

### Using Pre-trained Model

If you don't want to train the model from scratch, you can use the pre-trained model:

1. Locate the `obj.zip` archive in the project directory.
2. Unzip the `obj.zip` archive to retrieve the trained model (`model.pkl`) and CountVectorizer object (`vectorizer.pkl`).
3. Ensure these files are placed in the `objs/` directory.

Using the pre-trained model allows you to skip the model training step and start using the application immediately.

### Training the Model

If you choose to train the model yourself:

1. Ensure you have the training data. The model has been trained on the "Spam Emails" dataset, which can be found at:
   https://www.kaggle.com/datasets/noeyislearning/spam-emails/data

2. Download the dataset and save it as `emails.csv` in the project directory.

3. Run the model training script:
   ```
   python model_training.py
   ```
   This will create `model.pkl` and `vectorizer.pkl` in the `objs/` directory.

### Running the Application

Start the FastAPI application:
```
uvicorn email_triggers:app --reload
```

The application will be available at `http://127.0.0.1:8000`.

### API Endpoints

1. Check and classify new emails:
   ```
   GET /check_email
   ```
   This endpoint retrieves unread emails, classifies them, and updates labels in Gmail.

2. Classify a single email:
   ```
   POST /classify_email
   ```
   Request body:
   ```json
   {
     "subject": "Email subject",
     "body": "Email body text"
   }
   ```
   Response:
   ```json
   {
     "prediction": "spam" or "ham"
   }
   ```

### Testing

To test the Gmail IMAP connection:
```
python test.py
```
This script verifies the connection to your Gmail account using the provided credentials.

### Troubleshooting

1. IMAP Connection Issues:
   - Ensure your Gmail account has IMAP enabled in settings.
   - If using 2-factor authentication, generate an app password for this application.
   - Check your `.env` file for correct credentials.

2. Model Training Errors:
   - Verify `emails.csv` exists and has the correct format.
   - Ensure all required packages are installed.

3. API Errors:
   - Check the `email_classifier.log` file for detailed error messages.
   - Verify that `model.pkl` and `vectorizer.pkl` exist in the `objs/` directory.

### Performance Optimization

- Monitor the execution time of the `/check_email` endpoint, especially for large inboxes.
- Consider implementing batch processing for large numbers of emails.
- Optimize the `clean_text` function if processing speed becomes a bottleneck.

## Data Flow

The email classification system follows this data flow:

1. User triggers the `/check_email` endpoint.
2. Application connects to Gmail via IMAP.
3. Unread emails are fetched from the inbox.
4. Each email's subject and body are extracted and preprocessed.
5. The preprocessed text is vectorized using the pre-trained CountVectorizer.
6. The vectorized text is passed through the Naive Bayes classifier.
7. Based on the classification result, the email is labeled as spam or ham in Gmail.
8. Classification results are logged and returned to the user.

```
[User] -> [FastAPI App] -> [Gmail IMAP] -> [Email Extraction]
                                                |
                                                v
[Classification Result] <- [Naive Bayes Model] <- [Text Vectorization]
        |
        v
[Gmail Label Update] -> [Logging] -> [User Response]
```

Note: The model training process is separate and occurs offline before the application is deployed.