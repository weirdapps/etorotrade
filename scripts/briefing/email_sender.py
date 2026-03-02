"""Send HTML briefing email via Gmail API using OAuth."""

import base64
import json
import os
import tempfile
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


SCOPES = ['https://www.googleapis.com/auth/gmail.send']
DEFAULT_RECIPIENT = 'plessas@nbg.gr'
DEFAULT_SENDER = 'me'


def get_gmail_credentials():
    """Build Gmail credentials from environment variable or token file.

    The GMAIL_TOKEN env var should contain JSON with:
        client_id, client_secret, refresh_token
    """
    token_json = os.environ.get('GMAIL_TOKEN', '')

    if token_json.strip().startswith('{'):
        token_data = json.loads(token_json)
    else:
        # Try as file path
        token_path = token_json or os.path.expanduser(
            '~/.google-skills/gmail/gmail_token.json'
        )
        if not os.path.exists(token_path):
            raise FileNotFoundError(
                f"Gmail token not found at {token_path}. "
                "Set GMAIL_TOKEN env var with JSON content or file path."
            )
        with open(token_path, 'r') as f:
            token_data = json.load(f)

    creds = Credentials(
        token=None,
        refresh_token=token_data.get('refresh_token'),
        token_uri='https://oauth2.googleapis.com/token',
        client_id=token_data.get('client_id'),
        client_secret=token_data.get('client_secret'),
        scopes=SCOPES,
    )

    # Refresh the access token
    creds.refresh(Request())
    return creds


def send_email(html_content, subject, recipient=None):
    """Send an HTML email via Gmail API.

    Args:
        html_content: HTML string for the email body
        subject: Email subject line
        recipient: Email address (defaults to DEFAULT_RECIPIENT)
    """
    recipient = recipient or DEFAULT_RECIPIENT
    creds = get_gmail_credentials()
    service = build('gmail', 'v1', credentials=creds)

    message = MIMEText(html_content, 'html')
    message['to'] = recipient
    message['subject'] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    body = {'raw': raw}

    sent = service.users().messages().send(
        userId=DEFAULT_SENDER, body=body
    ).execute()

    print(f"Email sent successfully. Message ID: {sent.get('id')}")
    return sent
