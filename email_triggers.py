import os
from dotenv import load_dotenv
from model_training import clean_text
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import imaplib
import email
from email.header import decode_header
import base64
import joblib  # For loading your ML model
import logging
from datetime import datetime

# Initialize FastAPI app
load_dotenv()
app = FastAPI()

# Load your spam classification model
model = joblib.load('obj/model.pkl')
cv = joblib.load('obj/vectorizer.pkl')

# Email credentials (IMAP)
EMAIL = os.environ.get("EMAIL_ID")
PASSWORD = os.environ.get("PASSWORD")
IMAP_SERVER = "imap.gmail.com"

if not EMAIL or not PASSWORD:
    logging.error("Email credentials are not set in the environment variables.")
    logging.error("Email credentials are not set in the environment variables.")
    raise EnvironmentError("Email credentials are not set in the environment variables.")

# Setup logging
logging.basicConfig(
    handlers=[
        logging.FileHandler("email_classifier.log"),
        logging.StreamHandler()
    ],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Request model (optional, in case of email text input)
class EmailText(BaseModel):
    subject: str
    body: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Email Classifier API"}


@app.get("/check_email")
def check_email():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox")
        status, messages = mail.search(None, 'UNSEEN')
        logging.info(f"Fetched messages: {messages}")
        mail_ids = messages[0].split()
    except imaplib.IMAP4.error as e:
        logging.error(f"IMAP error: {e}")
        return {"error": "Failed to connect to the email server."}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"error": "An unexpected error occurred."}

    classified = {
        "spam": 0, 
        "ham": 0, 
        "spam_details": []  # List to store details of spam emails
    }
    
    for num in mail_ids:
        _, msg = mail.fetch(num, '(RFC822)')
        for response in msg:
            if isinstance(response, tuple):
                raw_email = response[1]
                msg = email.message_from_bytes(raw_email)
                
                try:
                    decoded_header = decode_header(msg["Subject"])
                    subject, encoding = decoded_header[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else 'utf-8')
                except Exception as e:
                    logging.error(f"Error decoding subject: {e}")
                    subject = "No Subject"

                body = extract_email_body(msg)
                sender = msg["From"]
                date_received = msg["Date"]
                
                email_data = EmailText(subject=subject, body=body)
                prediction = classify_email_endpoint(email_data)
                
                if prediction['prediction'] == 'spam':
                    mail.store(num, '+X-GM-LABELS', '\\Spam')
                    classified["spam"] += 1
                    spam_confidence = float(prediction['confidence'][0])
                    
                    # Store spam details
                    spam_info = {
                        "subject": subject,
                        "sender": sender,
                        "date": date_received,
                        "confidence": spam_confidence
                    }
                    classified["spam_details"].append(spam_info)
                    
                    # Log spam details
                    log_spam_email(subject, sender, date_received, spam_confidence)
                else:
                    classified["ham"] += 1
    
    return classified


@app.post("/classify_email")
def classify_email_endpoint(email: EmailText):
    try:
        combined_text = f"{email.subject} {email.body}"
        cleaned_text = clean_text(combined_text)
        vector = cv.transform([cleaned_text])
        prediction = model.predict_proba(vector)
        return {
            "prediction": "spam" if prediction[0][1] > 0.5 else "ham",
            "confidence": prediction[:,1]  # This will be a numpy array
        }
    except Exception as e:
        logging.error(f"Error during email classification: {e}")
        return {"error": "Failed to classify the email."}




def extract_email_body(msg):
    try:
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    return part.get_payload(decode=True).decode()
        else:
            return msg.get_payload(decode=True).decode()
    except Exception as e:
        logging.error(f"Error extracting email body: {e}")
        return ""


def log_spam_email(subject, sender, date_received, confidence):
    logging.info(
        f"Spam detected:\n"
        f"Subject: {subject}\n"
        f"From: {sender}\n"
        f"Date: {date_received}\n"
        f"Confidence: {confidence:.4f}"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)