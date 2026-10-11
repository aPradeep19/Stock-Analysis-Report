"""
Email Dispatcher 

Sends the newsletter through Gmail's SMTP server using only Python's standard library.

Design decisions:
1. One email, everyone BCC'd. The visible "To" is your own address and the recipients are
   passed to the SMTP server separately, so nobody on the list sees anyone else's address
   (and nobody can "Reply All" to the whole list).
2. multipart/alternative: an HTML version plus a plain-text version of the same report.
   Email apps pick the best one they can show, and spam filters trust mail that has both.
3. Attachments (the PDF) are optional, so the dispatcher works the same with or without one.
4. Uses email.message.EmailMessage, the modern replacement for the older
   MIMEMultipart/MIMEText classes. It builds the same MIME structure with less code.
5. send() returns True/False instead of crashing, and main.py turns False into a failed
   exit code. On GitHub Actions a failed run sends you a notification email, so a broken
   send is never silent.
"""

import mimetypes
import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr, make_msgid


def parse_recipients(raw: str) -> list:
    """
    Turns "a@x.com, b@y.com; c@z.com" (commas, semicolons, spaces or new lines) into a clean,
    de-duplicated list. Storing the list as ONE secret keeps addresses out of the public repo.
    """
    if not raw:
        return []
    for sep in (";", "\n", " "):
        raw = raw.replace(sep, ",")
    seen, recipients = set(), []
    for address in (part.strip() for part in raw.split(",")):
        if address and "@" in address and address.lower() not in seen:
            seen.add(address.lower())
            recipients.append(address)
    return recipients


class EmailDispatcher:
    def __init__(self, sender_email: str, app_password: str, recipients: list,
                 sender_name: str = "Pre-Market Brief",
                 smtp_server: str = "smtp.gmail.com", smtp_port: int = 465):
        self.sender_email = sender_email
        # Gmail shows app passwords as "abcd efgh ijkl mnop"; the spaces aren't part of it
        self.app_password = (app_password or "").replace(" ", "")
        self.recipients = recipients
        self.sender_name = sender_name
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def build_message(self, subject: str, html_body: str, text_body: str,
                      attachments: list = None) -> EmailMessage:
        """Assembles the email. Kept separate from send() so tests can inspect it without sending."""
        msg = EmailMessage()
        msg["From"] = formataddr((self.sender_name, self.sender_email))
        msg["To"] = self.sender_email  # recipients go in the SMTP envelope only (BCC)
        msg["Subject"] = subject
        msg["Message-ID"] = make_msgid(domain=self.sender_email.split("@")[-1])

        msg.set_content(text_body)                       # plain-text part
        msg.add_alternative(html_body, subtype="html")   # HTML part (preferred by most apps)

        for path in attachments or []:
            if not path or not os.path.exists(path):
                continue
            mime_type, _ = mimetypes.guess_type(path)
            maintype, subtype = (mime_type or "application/octet-stream").split("/", 1)
            with open(path, "rb") as f:
                msg.add_attachment(f.read(), maintype=maintype, subtype=subtype,
                                   filename=os.path.basename(path))
        return msg

    def send(self, subject: str, html_body: str, text_body: str, attachments: list = None) -> bool:
        if not self.recipients:
            print("[ERROR] No recipients configured. Set RECIPIENT_EMAILS.")
            return False

        msg = self.build_message(subject, html_body, text_body, attachments)
        envelope = [self.sender_email] + [r for r in self.recipients if r.lower() != self.sender_email.lower()]

        print(f"\n[*] Sending to {len(self.recipients)} recipient(s) via {self.smtp_server}...")
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context, timeout=30) as server:
                server.login(self.sender_email, self.app_password)
                refused = server.send_message(msg, from_addr=self.sender_email, to_addrs=envelope)
            if refused:
                # Gmail accepted the message but rejected some addresses (typos, closed inboxes)
                print(f"[WARN] Some recipients were refused: {', '.join(refused)}")
            print("[✓] Email delivered to Gmail for sending.")
            return True
        except smtplib.SMTPAuthenticationError:
            print("[ERROR] Gmail rejected the login. Check SENDER_EMAIL and that EMAIL_PASSWORD is a "
                  "16-character App Password (not your normal Gmail password).")
        except (smtplib.SMTPException, OSError) as e:
            print(f"[ERROR] Failed to send email: {e}")
        return False
