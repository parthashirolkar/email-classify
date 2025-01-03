import os
from multiprocessing import Pool
from typing import List, Tuple
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from model_training import clean_text
import imaplib
import email
from email.header import decode_header
import joblib
import logging

load_dotenv()


def load_assets() -> Tuple[MultinomialNB, CountVectorizer]:
    model = joblib.load('obj/model.pkl')
    cv = joblib.load('obj/vectorizer.pkl')
    return model, cv

model, cv = load_assets()

EMAIL = os.environ.get("EMAIL_ID")
PASSWORD = os.environ.get("PASSWORD")
IMAP_SERVER = "imap.gmail.com"

if not EMAIL or not PASSWORD:
    logging.error("Email credentials are not set in the environment variables.")
    raise EnvironmentError("Email credentials are not set in the environment variables.")


logging.basicConfig(
    handlers=[
        logging.FileHandler("email_classifier.log"),
        logging.StreamHandler()
    ],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class EmailText:
    def __init__(self, subject, body):
        self.subject = subject
        self.body = body

def check_email():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox", readonly=True)
        _, messages = mail.search(None, 'UNSEEN')
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
    
    email_texts = []
    email_data_list = []
    spam_ids = []
    
    if mail_ids:
        _, msgs = mail.fetch(','.join(id.decode() for id in mail_ids), '(RFC822)')
        for mail_id, response in zip(mail_ids, msgs):
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
                email_texts.append(f"{subject} {body}")
                
                email_data_list.append({
                    "num": mail_id.decode(),
                    "subject": subject,
                    "sender": msg["From"],
                    "date": msg["Date"]
                })
    
    if email_texts:
        try:
            with Pool() as pool:
                cleaned_texts = pool.map(clean_text, email_texts)
            vectors = cv.transform(cleaned_texts)
            predictions = model.predict_proba(vectors)
            
            for i, prediction in enumerate(predictions):
                email_data = email_data_list[i]
                if prediction[1] > 0.5:
                    spam_ids.append(email_data['num'])
                    classified["spam"] += 1
                    spam_info = {
                        "subject": email_data["subject"],
                        "sender": email_data["sender"],
                        "date": email_data["date"],
                        "confidence": prediction[1]
                    }
                    classified["spam_details"].append(spam_info)
                    log_spam_email(email_data['subject'], email_data['sender'], email_data['date'], prediction[1])
                else:
                    classified["ham"] += 1
            
            if spam_ids:
                mail.store(','.join(spam_ids), '+X-GM-LABELS', '\\Spam')
        except Exception as e:
            logging.error(f"Error during batch email classification: {e}")
            return {"error": "Failed during batch processing."}
    
    return classified

def classify_email(email):
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
    result = check_email()
    logging.info(f"Email classification result: {result}")
    