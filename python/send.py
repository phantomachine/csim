import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import socket
import os

""" Module for sending automated messages. Useful for batch jobs that take a long time.

Example Usage:

    import send

    # Custom message
    message = "Hey there sexy!"

    # Send an automatic email to subscribers: see SEND.PY
    send.message(rootdir, message \
                toaddr=["author1@gmail.com", "author2@gmail.com"])

Copyright 2017, T.Kam and J.Lee
"""

def message(rootdir=' ', custom_message="", toaddr=["arttuditoo@gmail.com"]):

    # Current machine name
    machine = socket.gethostbyaddr(socket.gethostname())[0]

    # Customisable items below ===============================================
    subject_title = custom_message + "Job (" + rootdir + ") on server: " \
                                                                + machine

    body = "\n" + custom_message \
            + "\nYour job " + rootdir + " on " + machine \
            + "\n\nView saved results under the directory:" \
            + "\n\n" + os.getcwd() + "/" + rootdir  \
            + "\n\nLove, \n Art Tuditoo"

    # Setup sender's login details (GMAIL)
    fromaddr = "arttuditoo@gmail.com"
    passwd = "ANewHope1977"
    server = smtplib.SMTP('smtp.gmail.com', 587)
    # Customisable items above ===============================================

    # Email MIME protocol
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr[0]
    # Customisable message
    msg['Subject'] = subject_title

    # Attach body content to message
    msg.attach(MIMEText(body, 'plain'))

    # Contact server and send
    server.starttls()
    server.login(fromaddr, passwd)
    text = msg.as_string()
    try:
        print("\nSending email to the following recipients:")
        for i, address in enumerate(toaddr):
            server.sendmail(fromaddr, address, text)
            print("\n\t %i. %s" % (i, address) )
        server.quit()
        print("\nEmail sent!")
    except:
        print("\nSomething went wrong with email sending ...")
