# Import smtplib for the actual sending function
import smtplib
from utils.config import process_config

def notify_email():
    # fetch config
    config = process_config("../configs/test_config.json")
    gmail_user = "aikaffe1992@gmail.com"
    gmail_password = 'Aikaffe23'

    sent_from = gmail_user
    to = [config.email_recipient]

    subject = "SC2AI Report"
    email_text = "Agent training on map %s have completed after %d episodes. Nice work dude." % (config.map_name,
                                                                                                 config.total_episodes)

    message = 'Subject: {}\n\n{}'.format(subject, email_text)

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, message)
        server.close()

        print('Email sent!')
    except:
        print('Could not send email')
