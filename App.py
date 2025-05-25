from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
import mysql.connector
import base64, os, sys
import nltk
from numba.cuda import selp

nltk.download('punkt')  # Ensure required data is downloaded

app = Flask(__name__)
app.secret_key = 'a'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/AdminLogin')
def AdminLogin():
    return render_template('AdminLogin.html')


@app.route('/OfficerLogin')
def OfficerLogin():
    return render_template('OfficerLogin.html')


@app.route('/UserLogin')
def UserLogin():
    return render_template('UserLogin.html')


@app.route('/NewOfficer')
def NewOfficer():
    return render_template('NewOfficer.html')


@app.route('/NewUser')
def NewUser():
    return render_template('NewUser.html')


@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    if request.method == 'POST':
        if request.form['uname'] == 'admin' and request.form['password'] == 'admin':

            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
            cur = conn.cursor()
            cur.execute("SELECT * FROM officertb ")
            data = cur.fetchall()

            return render_template('AdminHome.html', data=data)

        else:
            flash('Username or Password is wrong')
            return render_template('AdminLogin.html')


@app.route("/AdminHome")
def AdminHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM officertb ")
    data = cur.fetchall()

    return render_template('AdminHome.html', data=data)


@app.route('/Remove1')
def Remove1():
    id = request.args.get('id')
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
    cursor = conn.cursor()
    cursor.execute(
        "Delete from  officertb  where id='" + id + "'")
    conn.commit()
    conn.close()

    flash('Officer Remove Successfully ')
    return AdminHome()


@app.route("/AUserInfo")
def AUserInfo():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb ")
    data = cur.fetchall()
    return render_template('AUserInfo.html', data=data)



@app.route('/Remove2')
def Remove2():
    id = request.args.get('id')
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
    cursor = conn.cursor()
    cursor.execute(
        "Delete from  regtb  where id='" + id + "'")
    conn.commit()
    conn.close()

    flash('User Remove Successfully ')
    return AUserInfo()

@app.route('/AComplaintInfo')
def AComplaintInfo():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM complainttb ORDER BY id ")
    data1 = cur.fetchall()
    return render_template('AComplaintInfo.html', data=data1)


@app.route("/newofficer", methods=['GET', 'POST'])
def newofficer():
    if request.method == 'POST':
        uname = request.form['uname']
        mobile = request.form['mobile']
        email = request.form['email']
        address = request.form['address']
        username = request.form['username']
        password = request.form['password']
        depart = request.form['depart']
        dist = request.form['dist']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from officertb where username='" + username + "'  ")
        data = cursor.fetchone()
        if data is None:
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO officertb VALUES ('','" + uname + "','" + mobile + "','" + email + "','" + address + "','" +
                depart + "','" + username + "','" + password + "','" + dist + "')")
            conn.commit()
            conn.close()

            flash('Record Saved!')
            return render_template('NewOfficer.html')
        else:
            flash('Already Register This  Officer Name!')
            return render_template('NewOwner.html')


@app.route("/officerlogin", methods=['GET', 'POST'])
def officerlogin():
    if request.method == 'POST':

        username = request.form['uname']
        password = request.form['password']
        session['oname'] = request.form['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from officertb where username='" + username + "' and Password='" + password + "' ")
        data = cursor.fetchone()
        if data is None:

            flash('Username or Password is wrong')
            return render_template('OfficerLogin.html')

        else:

            session['city'] = data[8]
            conn = mysql.connector.connect(user='root', password='', host='localhost',
                                           database='1epetitiondb')
            cur = conn.cursor()
            cur.execute("SELECT * FROM officertb where username='" + session['oname'] + "'")
            data1 = cur.fetchall()
            return render_template('OfficerHome.html', data=data1)


@app.route('/OfficerHome')
def OfficerHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM officertb where username='" + session['oname'] + "'")
    data1 = cur.fetchall()
    return render_template('OfficerHome.html', data=data1)


@app.route('/OUserInfo')
def OUserInfo():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb")
    data = cur.fetchall()

    return render_template('OUserInfo.html', data=data)


@app.route('/OComplaintInfo')
def OComplaintInfo():
    oname = session['oname']
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
    cursor = conn.cursor()
    cursor.execute("SELECT * from officertb where username='" + oname + "'")
    data = cursor.fetchone()
    if data:
        dep = data[5]
        print(dep)
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM complainttb where Type='" + dep + "' and Status='Waiting' and City='"+ session['city']  +"'")
    data1 = cur.fetchall()
    return render_template('OComplaintInfo.html', data=data1)


@app.route('/OUpdate')
def OUpdate():
    oname = session['oname']
    id = request.args.get('id')
    uname = request.args.get('uname')
    session['cid'] = id
    session['uname'] = uname
    return render_template('OUpdate.html', oname=oname)


@app.route("/oupdate", methods=['GET', 'POST'])
def oupdate():
    if request.method == 'POST':
        id = session['cid']
        oname = session['oname']

        Answer = request.form['Answer']

        file = request.files['file']
        import random
        fnew = random.randint(111, 999)
        savename = str(fnew) + file.filename
        file.save("static/upload/" + savename)

        import datetime
        date = datetime.datetime.now().strftime('%Y-%m-%d')

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
        cursor = conn.cursor()
        cursor.execute(
            "update complainttb set Status='Completed',Ans='" + Answer + "',Image1='" + savename + "',OfficerName='" + oname + "' where id='" + id + "'")
        conn.commit()
        conn.close()

        flash('Complaint Updated Successfully ')

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + session['uname'] + "'  ")
        data = cursor.fetchone()
        if data:
            mail = data[3]
            mob = data[2]
            sendmail(mail, "Complaint action Info" + Answer)
            sendmail(mob, "Complaint action Info" + Answer)

        return render_template('OUpdate.html', oname=oname)


def sendmsg(targetno, message):
    import requests
    requests.post(
        "http://sms.creativepoint.in/api/push.json?apikey=6555c521622c1&route=transsms&sender=FSSMSS&mobileno=" + targetno + "&text=Dear customer your msg is " + message + "  Sent By FSMSG FSSMSS")


@app.route("/newuser", methods=['GET', 'POST'])
def newuser():
    if request.method == 'POST':
        uname = request.form['uname']
        mobile = request.form['mobile']
        email = request.form['email']
        address = request.form['address']
        username = request.form['username']
        password = request.form['password']
        dist = request.form['dist']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + username + "'  ")
        data = cursor.fetchone()
        if data is None:
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO regtb VALUES ('','" + uname + "','" + mobile + "','" + email + "','" + address + "','" +
                username + "','" + password + "','" + dist + "')")
            conn.commit()
            conn.close()

            flash('Record Saved!')

            return render_template('NewUser.html')
        else:
            flash('Already Register This  UserName!')
            return render_template('NewUser.html')


@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():
    if request.method == 'POST':

        username = request.form['uname']
        password = request.form['password']

        session['uname'] = request.form['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + username + "' and Password='" + password + "' ")
        data = cursor.fetchone()
        if data is None:

            flash('Username or Password is wrong')
            return render_template('UserLogin.html')

        else:
            session['city'] = data[7]
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb where UserName='" + session['uname'] + "'")
            data1 = cur.fetchall()
            return render_template('UserHome.html', data=data1)


@app.route('/UserHome')
def UserHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb where UserName='" + session['uname'] + "'")
    data1 = cur.fetchall()
    return render_template('UserHome.html', data=data1)


@app.route('/NewComplaint')
def NewComplaint():
    oname = session['uname']
    return render_template('NewComplaint.html', uname=oname)


@app.route("/owfileupload", methods=['GET', 'POST'])
def owfileupload():
    if request.method == 'POST':

        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        # Ensure required NLTK data is available
        nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()

        # Function to classify complaints based on keywords
        def classify_complaint(complaint):
            categories = {
                "Transportation and Traffic": ["vehicle", "roadblock", "traffic jam", "over speeding",  "commute", "route", "highway",
                                               "signal", "accident", "Bus not stopping", "road safety", "Public Transport Delay", "transportation"],
                "Public Health Related": ["health care", "wellness", "health policy", "mental health",
                                          "infection", "treatment", "hospital", "health insurance"],
                "Municipality": ["government", "garbage", "sewage", "drainage", "public toilet", "solid waste", "Trash collection", "pest control",  "governance", "municipal law", "waste management",
                                 "fire department", "police department", "housing", "social services"],
                "Urban Development": ["land use", "smart city", "sustainable development", "green spaces",
                                      "building permits", "urbanization", "environmental impact"],
                "Public Work Department": ["road maintenance", "bridge damage", "road blockage", "traffic signal", "culvert", "road marking", "", "", "water supply", "sewage system",
                                           "drainage", "street lighting"]
            }

            complaint_lower = complaint.lower()  # Convert complaint to lowercase once

            for category, keywords in categories.items():
                for keyword in keywords:
                    if keyword in complaint_lower:  # Match substrings including phrases
                        return category
            return "Uncategorized"

        oname = session['uname']

        loca = request.form['loca']
        info = request.form['info']
        savename = ""

        category = classify_complaint(info)

        if category == "Uncategorized":
            flash('Complaint Not  Upload Keyword Not Match ')
            return render_template('NewComplaint.html', oname=oname)

        # Analyze sentiment
        sentiment_scores = sia.polarity_scores(info)
        sentiment = "Positive" if sentiment_scores['compound'] >= 0.05 else "Negative" if sentiment_scores[
                                                                                              'compound'] <= -0.05 else "Neutral"

        import datetime
        date = datetime.datetime.now().strftime('%Y-%m-%d')

        conn = mysql.connector.connect(user='root', password='', host='localhost',
                                       database='1epetitiondb')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO complainttb VALUES ('','" + oname + "','" + loca + "','" + info + "','" + savename + "','" + date + "','Waiting','" + category + "','','','','"+ session['city']+"')")
        conn.commit()
        conn.close()
        flash('Complaint  Upload Successfully ')

        return render_template('NewComplaint.html', oname=oname)


@app.route('/NewComplaint1')
def NewComplaint1():
    oname = session['uname']
    return render_template('NewComplaint1.html', uname=oname)


"""
@app.route("/owfileupload1", methods=['GET', 'POST'])
def owfileupload1():
    if request.method == 'POST':

        oname = session['uname']

        loca = request.form['loca']
        info = request.form['info']
        import tensorflow as tf
        import cv2
        import sys
        import numpy as np
        from keras.preprocessing import image

        file = request.files['file']
        import random
        fnew = random.randint(111, 999)
        savename = str(fnew) + file.filename
        #file.save("static/upload/" + savename)

        img1 = cv2.imread('static/upload/'+savename)
        dst = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
        noi = 'static/upload/noi.png'

        cv2.imwrite(noi, dst)

        import warnings
        warnings.filterwarnings('ignore')

        model = tf.keras.models.load_model('model.h5')
        test_image = image.load_img('static/upload/{savename}', target_size=(200, 200))
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
        print(result)
        ind = np.argmax(result)
        out = ''
        pre = ""

        if ind == 0:
            out = 'pathole'
            print(out)
            pre = 'Municipality'

        elif ind == 1:
            out = 'water'
            print(out)
            pre = 'Public Work Department'


        category = pre

        # Analyze sentiment

        import datetime
        date = datetime.datetime.now().strftime('%Y-%m-%d')

        conn = mysql.connector.connect(user='root', password='', host='localhost',
                                       database='1epetitiondb')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO complainttb VALUES ('','" + oname + "','"+ loca +"','" + info + "','" + savename + "','" + date + "','Waiting','"+ category +"','','','')")
        conn.commit()
        conn.close()
        flash('File Upload Successfully ')

        return render_template('NewComplaint.html', oname=oname)
"""


@app.route("/owfileupload1", methods=['GET', 'POST'])
def owfileupload1():
    if request.method == 'POST':
        oname = session['uname']

        loca = request.form['loca']
        info = "Nil"

        import tensorflow as tf
        import cv2
        import sys
        import numpy as np
        from keras.preprocessing import image
        import random

        file = request.files['file']
        fnew = random.randint(111, 999)
        savename = str(fnew) + file.filename
        file_path = "static/upload/" + savename

        # Save the uploaded file
        file.save(file_path)

        # Read the saved image
        img1 = cv2.imread(file_path)

        # Check if the image is loaded correctly
        if img1 is None:
            raise ValueError(f"Image not found or not loaded properly: {file_path}")

        # Ensure image is in 3-channel format
        if len(img1.shape) == 2:  # Grayscale image
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

        # Apply noise reduction
        dst = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
        noi = 'static/upload/noi.png'
        cv2.imwrite(noi, dst)

        # Load model and predict
        model = tf.keras.models.load_model('model.h5')
        img = image.load_img(file_path, target_size=(200, 200))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize

        prediction = model.predict(img_array)
        ind = np.argmax(prediction)
        print(ind)

        if ind == 0:
            category = 'Municipality'
        else:
            category = 'Public Work Department'

        # Store in database
        import datetime
        date = datetime.datetime.now().strftime('%Y-%m-%d')

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO complainttb VALUES ('', %s, %s, %s, %s, %s, 'Waiting', %s, '', '', '',%s)",
            (oname, loca, info, savename, date, category,session['city']))
        conn.commit()
        conn.close()

        flash('File Upload Successfully')
        return render_template('NewComplaint.html', oname=oname)


@app.route("/UComplaintInfoInfo")
def UComplaintInfoInfo():
    uname = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM complainttb where UserName='" + uname + "'")
    data = cur.fetchall()

    return render_template('UComplaintInfoInfo.html', data=data)


@app.route('/Remove')
def Remove():
    id = request.args.get('id')
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1epetitiondb')
    cursor = conn.cursor()
    cursor.execute(
        "Delete from  complainttb  where id='" + id + "'")
    conn.commit()
    conn.close()

    flash('Complaint Remove Successfully ')
    return UComplaintInfoInfo()


def sendmail(Mailid, message):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    fromaddr = "projectmailm@gmail.com"
    toaddr = Mailid

    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = fromaddr

    # storing the receivers email address
    msg['To'] = toaddr

    # storing the subject
    msg['Subject'] = "Alert"

    # string to store the body of the mail
    body = message

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)

    # start TLS for security
    s.starttls()

    # Authentication
    s.login(fromaddr, "qmgn xecl bkqv musr")

    # Converts the Multipart msg into a string
    text = msg.as_string()

    # sending the mail
    s.sendmail(fromaddr, toaddr, text)

    # terminating the session
    s.quit()


if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True, port=5000)
    app.run(debug=True, use_reloader=True)
