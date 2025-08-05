# Merged Flask App: Binary + Multiclass Fraud Detection
import json
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from flask_mail import Mail, Message
from flask_bcrypt import Bcrypt
import pandas as pd
import pickle
import os
from datetime import datetime
import traceback
# import random # No longer explicitly needed for random choice as model outputs specific type

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_and_random_key_here_12345')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///transactions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email config
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
# IMPORTANT: Replace with environment variables or actual credentials
app.config['MAIL_USERNAME'] = os.environ.get('EMAIL_USER', 'your_email@gmail.com') # <<<-- SET THIS
app.config['MAIL_PASSWORD'] = os.environ.get('EMAIL_PASS', 'your_email_password') # <<<-- SET THIS
ALERT_RECIPIENT_EMAIL = os.environ.get('ALERT_EMAIL', 'recipient@example.com') # <<<-- SET THIS

# Init extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
bcrypt = Bcrypt(app)
mail = Mail(app)

# Load models and encoders
# Ensure these .pkl files are in the same directory as app.py or provide full paths
ML_MODEL = None
LB_LOCATION = None
FRAUD_TYPE_MODEL = None
LABEL_ENCODER_FTYPE = None

try:
    ML_MODEL = pickle.load(open("fraud-svc-model.pkl", "rb"))
    LB_LOCATION = pickle.load(open("location-lb.pkl", "rb"))
    FRAUD_TYPE_MODEL = pickle.load(open("fraud-dt-model.pkl", "rb"))
    LABEL_ENCODER_FTYPE = pickle.load(open("Ftype-lb.pkl", "rb"))
    print("All models and encoders loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Model or encoder file not found: {e}. Please ensure all .pkl files are in the correct directory.")
    print("Application may not function correctly without models.")
    # In a production environment, you might want to exit or log a critical error here.
except Exception as e:
    print(f"ERROR: An unexpected error occurred while loading models: {e}")
    print("Application may not function correctly without models.")

# Define predefined locations for the dropdown
# This list must precisely match the categories used during the training of your models
PREDEFINED_LOCATIONS = [
    'Bangalore', 'Surat', 'Hyderabad', 'Mumbai', 'Kolkata',
    'Ahmedabad', 'NaN', 'Jaipur', 'Delhi', 'Chennai', 'Pune'
]

# --- NEW/UPDATED: Fraud Type Information Dictionary ---
FRAUD_TYPES_INFO = {
    "scam": {
        "title": "Scam Alert!",
        "description": "Scams involve deceptive schemes to trick individuals into parting with money or personal information, often through false promises or urgent requests.",
        "how_it_is_done": [
            "**Phony Investment Opportunities:** Promising high returns with little to no risk.",
            "**Lottery/Prize Scams:** Notifying you that you've won a lottery or prize you never entered, asking for an upfront fee to claim it.",
            "**Romance Scams:** Building a fake romantic relationship online to gain trust and ask for money.",
            "**Emergency Scams (Grandparent Scams):** Posing as a family member in distress needing immediate funds.",
            "**Tech Support Scams:** Pretending to be tech support, claiming your computer has a virus, and demanding payment for unnecessary services."
        ],
        "tips_to_solve": [
            "**Verify Identity:** Always verify the identity of the person or organization contacting you, especially if they're asking for money or sensitive information. Use official contact methods, not those provided by the caller/sender.",
            "**Be Skeptical of Unsolicited Offers:** If it sounds too good to be true, it probably is.",
            "**Never Pay Upfront Fees:** Legitimate lotteries or prizes don't ask for money to release winnings.",
            "**Protect Personal Information:** Be cautious about sharing personal or financial details online or over the phone.",
            "**Use Strong Passwords & 2FA:** Protect your accounts. Enable two-factor authentication wherever possible.",
            "**Report Suspicious Activity:** Report scams to your local law enforcement, bank, or relevant consumer protection agencies."
        ]
    },
    "phishing": {
        "title": "Phishing Detected!",
        "description": "Phishing is a cybercrime where attackers disguise themselves as trustworthy entities (like banks, popular websites, or government agencies) in emails, text messages, or calls to trick individuals into revealing sensitive information, such as usernames, passwords, and credit card details.",
        "how_it_is_done": [
            "**Email Phishing:** Sending emails that look legitimate, containing malicious links or attachments.",
            "**Spear Phishing:** Targeting specific individuals with personalized phishing attempts.",
            "**Smishing (SMS Phishing):** Sending fraudulent text messages to trick recipients into clicking links or giving information.",
            "**Vishing (Voice Phishing):** Using phone calls to impersonate legitimate organizations and solicit information.",
            "**Website Spoofing:** Creating fake websites that mimic legitimate ones to capture credentials."
        ],
        "tips_to_solve": [
            "**Check the Sender:** Always verify the sender's email address or phone number. Look for inconsistencies or misspellings.",
            "**Hover Before Clicking:** Before clicking on any link, hover your mouse over it to see the actual URL. If it looks suspicious, don't click.",
            "**Look for Red Flags:** Be wary of urgent language, grammatical errors, generic greetings, or requests for sensitive information.",
            "**Don't Open Suspicious Attachments:** Malicious attachments can install malware.",
            "**Use Multi-Factor Authentication (MFA):** MFA adds an extra layer of security, making it harder for attackers to access your accounts even if they steal your password.",
            "**Keep Software Updated:** Regularly update your operating system, web browser, and security software.",
            "**Report Phishing Attempts:** Forward suspicious emails to your email provider or relevant cybersecurity agencies."
        ]
    },
    "Identity theft": { # Key matches the exact string from your list
        "title": "Identity Theft Risk!",
        "description": "Identity theft occurs when someone uses another person's personal identifying information, like their name, Social Security number, or credit card number, without their permission, to commit fraud or other crimes.",
        "how_it_is_done": [
            "**Data Breaches:** Criminals steal personal information from company databases.",
            "**Mail Theft:** Stealing mail containing bank statements, credit card offers, or other personal documents.",
            "**Phishing/Scams:** Tricking individuals into revealing personal information through deceptive tactics.",
            "**Skimming:** Using devices to steal credit/debit card information at ATMs or POS terminals.",
            "**Shoulder Surfing:** Looking over someone's shoulder to steal PINs or passwords.",
            "**Public Wi-Fi Exploitation:** Intercepting data on unsecured public Wi-Fi networks."
        ],
        "tips_to_solve": [
            "**Shred Sensitive Documents:** Dispose of documents containing personal information securely.",
            "**Monitor Financial Statements:** Regularly check bank and credit card statements for unauthorized activity.",
            "**Check Your Credit Report:** Obtain free annual credit reports to look for suspicious accounts or inquiries.",
            "**Create Strong, Unique Passwords:** Use a password manager and enable two-factor authentication (2FA) on all accounts.",
            "**Be Wary of Unsolicited Communications:** Don't provide personal information to unknown callers, emails, or texts.",
            "**Secure Your Mail:** Use a locking mailbox or pick up mail promptly.",
            "**Secure Your Devices:** Use antivirus software, firewalls, and keep your operating system updated.",
            "**Be Careful on Public Wi-Fi:** Avoid conducting financial transactions or sharing sensitive data on unsecured public networks.",
            "**Freeze Your Credit:** If you're highly concerned, consider freezing your credit with the major credit bureaus."
        ]
    },
    "Malware": { # Key matches the exact string from your list
        "title": "Malware Threat Detected!",
        "description": "Malware (malicious software) is any software intentionally designed to cause damage to a computer, server, client, or computer network, or to gather information without the owner's consent.",
        "how_it_is_done": [
            "**Phishing Emails:** Malware often arrives as an attachment in phishing emails or through malicious links.",
            "**Drive-by Downloads:** Visiting a compromised website can automatically download malware without your interaction.",
            "**Infected USB Drives:** Connecting an infected USB drive to your computer.",
            "**Software Vulnerabilities:** Exploiting unpatched security flaws in operating systems or applications.",
            "**Bundled Software:** Malware can be disguised as legitimate software or bundled with free downloads."
        ],
        "tips_to_solve": [
            "**Use Antivirus/Anti-Malware Software:** Install and keep reputable security software updated.",
            "**Keep Software Updated:** Regularly update your operating system, web browser, and all applications to patch vulnerabilities.",
            "**Be Cautious with Emails & Links:** Do not open suspicious emails, click on unknown links, or download attachments from untrusted sources.",
            "**Use a Firewall:** A firewall monitors network traffic and can block unauthorized access.",
            "**Back Up Your Data:** Regularly back up important files to an external drive or cloud storage.",
            "**Be Wary of Free Software/Downloads:** Free software can sometimes come bundled with unwanted malware.",
            "**Educate Yourself:** Stay informed about the latest malware threats and how they operate."
        ]
    },
    "Payment card fraud": { # Key matches the exact string from your list
        "title": "Payment Card Fraud Risk!",
        "description": "Payment card fraud involves the unauthorized use of a debit or credit card to obtain money or property. This includes fraudulent online purchases, physical card skimming, or using stolen card details.",
        "how_it_is_done": [
            "**Skimming:** Devices attached to card readers (ATMs, POS terminals) to steal card data.",
            "**Phishing/Vishing:** Tricking individuals into revealing card details through fake emails, calls, or texts.",
            "**Data Breaches:** Criminals hack into databases of merchants or financial institutions to steal card numbers.",
            "**Lost/Stolen Cards:** Using physical cards that have been lost or stolen.",
            "**Cloning:** Creating duplicate cards using stolen data.",
            "**Online Shopping Fraud:** Using stolen card details for unauthorized online purchases."
        ],
        "tips_to_solve": [
            "**Monitor Statements Regularly:** Check your bank and credit card statements frequently for unauthorized transactions.",
            "**Enable Transaction Alerts:** Set up SMS or email alerts for every transaction on your cards.",
            "**Protect Your PIN:** Never share your PIN and be discreet when entering it at ATMs or POS terminals.",
            "**Inspect Card Readers:** Before swiping or inserting your card, check card readers for any suspicious attachments.",
            "**Shop Securely Online:** Only make purchases on reputable websites with 'https://' in the URL and a padlock icon.",
            "**Be Wary of Public Wi-Fi for Transactions:** Avoid making online purchases or banking over unsecured public Wi-Fi.",
            "**Report Lost/Stolen Cards Immediately:** Contact your bank or card issuer as soon as you realize your card is missing.",
            "**Shred Card Offers:** Securely dispose of old credit card statements and unsolicited card offers."
        ]
    },
    # Generic fraud info if the specific type isn't found or model output is 'Not Available'
    "Not Available": {
        "title": "Potential Fraud Detected!",
        "description": "This transaction shows characteristics common to various types of financial fraud. The specific fraud type could not be identified at this time. It's crucial to take immediate action to protect your finances and personal information.",
        "how_it_is_done": [
            "Fraudsters use various sophisticated methods, including impersonation, deceptive offers, and exploiting vulnerabilities.",
            "Common tactics include unauthorized purchases, identity theft attempts, and tricking victims into sending money or revealing sensitive data."
        ],
        "tips_to_solve": [
            "**Review Your Accounts:** Immediately check all your bank accounts, credit cards, and online financial platforms for any suspicious activity.",
            "**Contact Your Bank/Card Issuer:** Report the suspicious transaction to your financial institution immediately.",
            "**Change Passwords:** Update passwords for all critical online accounts (email, banking, social media).",
            "**Enable Two-Factor Authentication (2FA):** Add an extra layer of security to your accounts.",
            "**Be Vigilant:** Be suspicious of unsolicited calls, emails, or messages asking for personal information or urgent actions.",
            "**Report the Incident:** File a report with your local law enforcement or relevant cybersecurity authorities."
        ]
    },
    "generic_fraud": { # Fallback if `output['fraud_type']` is truly unexpected
        "title": "Potential Fraud Detected!",
        "description": "This transaction shows characteristics common to various types of financial fraud. It's crucial to take immediate action to protect your finances and personal information.",
        "how_it_is_done": [
            "Fraudsters use various sophisticated methods, including impersonation, deceptive offers, and exploiting vulnerabilities.",
            "Common tactics include unauthorized purchases, identity theft attempts, and tricking victims into sending money or revealing sensitive data."
        ],
        "tips_to_solve": [
            "**Review Your Accounts:** Immediately check all your bank accounts, credit cards, and online financial platforms for any suspicious activity.",
            "**Contact Your Bank/Card Issuer:** Report the suspicious transaction to your financial institution immediately.",
            "**Change Passwords:** Update passwords for all critical online accounts (email, banking, social media).",
            "**Enable Two-Factor Authentication (2FA):** Add an extra layer of security to your accounts.",
            "**Be Vigilant:** Be suspicious of unsolicited calls, emails, or messages asking for personal information or urgent actions.",
            "**Report the Incident:** File a report with your local law enforcement or relevant cybersecurity authorities."
        ]
    }
}


# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    transactions = db.relationship('Transaction', backref='user', lazy=True)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    card_type = db.Column(db.String(50), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    purchase_category = db.Column(db.String(50), nullable=False)
    customer_age = db.Column(db.Integer, nullable=False)
    hour = db.Column(db.Integer, nullable=False)
    day = db.Column(db.Integer, nullable=False)
    weekday = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    result = db.Column(db.String(50), nullable=False)
    fraud_type = db.Column(db.String(100), nullable=False, default="Not Available")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Define the common feature columns that your models expect after encoding
# This list MUST match the column names your models were trained with
# (e.g., if 'card_type' was encoded to a number but the column was still named 'card_type')
MODEL_FEATURE_COLUMNS = [
    "amount", "card_type", "location", "purchase_category",
    "customer_age", "hour", "day", "weekday", "month"
]

# Helper function to preprocess input data for models
def _preprocess_data(amount, card_type, location, purchase_category, customer_age, hour, day, weekday, month):
    # Ensure models and encoders are loaded
    if LB_LOCATION is None:
        raise RuntimeError("Location LabelBinarizer not loaded.")

    # Encode categorical features
    # Ensure consistency with how your models were trained (e.g., mapping unknown to 0)
    encoded_card_type = {"mastercard": 1, "rupay": 2, "visa": 3}.get(str(card_type).lower(), 0)
    # Use .transform and handle potential unseen categories
    encoded_location = 0 # Default for unseen location
    if location in LB_LOCATION.classes_:
        encoded_location = LB_LOCATION.transform([location])[0]
    else:
        # If 'NaN' is explicitly a class in your LB_LOCATION, handle it.
        # Otherwise, 0 might be a reasonable default for unseen/missing.
        if 'NaN' in LB_LOCATION.classes_:
            encoded_location = LB_LOCATION.transform(['NaN'])[0]
        print(f"Warning: Location '{location}' not seen during training. Mapping to default encoded value.")

    encoded_purchase_category = {"digital": 1, "pos": 2}.get(str(purchase_category).lower(), 0)

    # Create a list of preprocessed numerical features
    input_features_list = [
        float(amount),
        encoded_card_type,
        encoded_location,
        encoded_purchase_category,
        float(customer_age),
        float(hour),
        float(day),
        float(weekday),
        float(month)
    ]
    
    # Create a DataFrame with the preprocessed data and correct column names
    df_processed = pd.DataFrame([input_features_list], columns=MODEL_FEATURE_COLUMNS)
    print(f"\nDataFrame for prediction:\n{df_processed}")
    print(f"DataFrame columns: {df_processed.columns.tolist()}")
    return df_processed


# Main prediction logic combining both models
def predict_fraud(transaction_data):
    if ML_MODEL is None or FRAUD_TYPE_MODEL is None or LABEL_ENCODER_FTYPE is None:
        raise RuntimeError("One or more ML models/encoders are not loaded. Cannot perform prediction.")

    # Preprocess the data once for both models
    df_for_prediction = _preprocess_data(**transaction_data)

    # Predict with the binary fraud detection model (ML_MODEL)
    is_fraud = ML_MODEL.predict(df_for_prediction)[0]
    

    fraud_type_result = "Not Applicable" # Default for non-fraudulent
    if is_fraud == 1:
        try:
            print(f"Attempting multiclass prediction for fraud type...")
            # If binary model detects fraud, then predict the type of fraud using FRAUD_TYPE_MODEL
            pred_encoded_fraud_type = FRAUD_TYPE_MODEL.predict(df_for_prediction)[0]
            print(f"Raw encoded fraud type prediction: {pred_encoded_fraud_type}")
            print(f"LABEL_ENCODER_FTYPE classes: {LABEL_ENCODER_FTYPE.classes_}")
            
            # Inverse transform to get the string representation of the fraud type
            # Ensure the inverse_transform can handle the predicted value
            if pred_encoded_fraud_type in range(len(LABEL_ENCODER_FTYPE.classes_)):
                fraud_type_result = LABEL_ENCODER_FTYPE.inverse_transform([pred_encoded_fraud_type])[0]
            else:
                fraud_type_result = "Not Available" # Fallback if inverse_transform fails for unseen/invalid encoding
                print(f"Warning: Predicted encoded fraud type {pred_encoded_fraud_type} is out of bounds for Ftype-lb.pkl classes.")

            print(f"Decoded fraud type prediction: {fraud_type_result}")

        except Exception as e:
            print(f"\n--- Multiclass prediction FAILED for fraud type ---")
            print(f"Error: {e}")
            print(f"Full Traceback:\n{traceback.format_exc()}")
            print(f"--- END Multiclass prediction FAILED ---\n")
            fraud_type_result = "Not Available" # Fallback if multiclass prediction fails
    
    # Determine the reason string
    reason_text = "Fraud Detected" if is_fraud else "Transaction is Safe"

    return {
        "is_fraudulent": int(is_fraud),
        "fraud_type": fraud_type_result, # This is the string like "scam", "Malware" etc.
        "reason": reason_text
    }

def send_fraud_alert_email(data, fraud_type):
    if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD'] or not ALERT_RECIPIENT_EMAIL:
        print("Email configuration is incomplete (MAIL_USERNAME, MAIL_PASSWORD, or ALERT_RECIPIENT_EMAIL not set). Skipping fraud alert email.")
        return
    try:
        msg = Message(
            "Fraud Alert: Urgent Action Required!",
            sender=app.config['MAIL_USERNAME'],
            recipients=[ALERT_RECIPIENT_EMAIL]
        )
        msg.body = f"""
        A potentially fraudulent transaction has been detected by FraudDetectPro!

        Fraud Type: {fraud_type}

        Transaction Details:
        Amount: {data.get('amount', 'N/A')}
        Card Type: {data.get('card_type', 'N/A')}
        Location: {data.get('location', 'N/A')}
        Purchase Category: {data.get('purchase_category', 'N/A')}
        Customer Age: {data.get('customer_age', 'N/A')}
        Timestamp (Hour/Day/Weekday/Month): {data.get('hour', 'N/A')}/{data.get('day', 'N/A')}/{data.get('weekday', 'N/A')}/{data.get('month', 'N/A')}

        Please investigate this transaction immediately.

        FraudDetectPro Team
        """
        mail.send(msg)
        print("Fraud alert email sent successfully!")
    except Exception as e:
        print(f"Failed to send fraud alert email: {e}")
        # Log full traceback here for debugging mail issues

# Routes
@app.route('/')
@login_required
def index():
    txns = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.timestamp.desc()).all()
    fraud_count = sum(1 for t in txns if t.result == 'Fraud')
    safe_count = sum(1 for t in txns if t.result == 'Safe')
    chart_data = { 'labels': ['Fraud', 'Safe'], 'data': [fraud_count, safe_count] }
    return render_template("index.html", fraud_safe_chart_data=chart_data, txns=txns)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    prediction_result = None # Initialize prediction_result for GET requests
    fraud_info = None # Initialize fraud_info for GET requests or non-fraud predictions

    if request.method == 'POST':
        form = request.form
        try:
            # Safely get form data, converting to expected types
            data = {
                'amount': float(form.get('amount')),
                'card_type': form.get('card_type'),
                'location': form.get('location'),
                'purchase_category': form.get('purchase_category'),
                'customer_age': float(form.get('customer_age')),
                'hour': int(form.get('hour')),
                'day': int(form.get('day')),
                'weekday': int(form.get('weekday')),
                'month': int(form.get('month'))
            }

            # Perform prediction
            output = predict_fraud(data)
            prediction_result = output # This now contains 'is_fraudulent', 'fraud_type', 'reason'
            
            # Save transaction to DB
            txn = Transaction(user_id=current_user.id, **data,
                              result="Fraud" if output['is_fraudulent'] else "Safe",
                              fraud_type=output['fraud_type']) # Store the specific fraud type
            db.session.add(txn)
            db.session.commit()

            if output['is_fraudulent']:
                send_fraud_alert_email(data, output['fraud_type'])
                flash(f"Fraud Detected! Type: {output['fraud_type']}", "danger")
                
                # --- NEW: Fetch and pass specific fraud info ---
                # Use the fraud_type directly from the model's output
                # Use .get() with a default in case the predicted type doesn't exist as a key
                fraud_info = FRAUD_TYPES_INFO.get(output['fraud_type'], FRAUD_TYPES_INFO["generic_fraud"])
                print(f"Serving fraud info for type: {output['fraud_type']}")
            else:
                flash("Transaction is safe.", "success")
                # No specific fraud info needed for safe transactions, so fraud_info remains None
                
        except ValueError as e:
            flash(f"Input Error: {e}. Please ensure all fields are correctly filled.", "danger")
            print(f"Input Error in /predict route: {e}")
        except RuntimeError as e: # Catch errors specifically from model loading/prediction issues
            flash(f"Prediction System Error: {e}. Please contact support.", "danger")
            print(f"Prediction system runtime error: {e}")
            print(f"Full Traceback for RuntimeError:\n{traceback.format_exc()}")
        except Exception as e:
            flash(f"An unexpected error occurred: {e}", "danger")
            print(f"Unhandled exception in /predict route: {e}") # For server-side debugging
            print(f"Full Traceback for unhandled exception:\n{traceback.format_exc()}")

    # Pass the predefined locations and prediction results/info to the template
    return render_template("predict.html", 
                           prediction=prediction_result, 
                           locations=PREDEFINED_LOCATIONS,
                           fraud_info=fraud_info # Pass the fraud_info dictionary
                          )

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            login_user(user)
            flash("Logged in successfully!", "success")
            return redirect(url_for('index'))
        flash("Invalid email or password", "danger")
    return render_template("login.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash("Username already exists", "danger")
        elif User.query.filter_by(email=email).first():
            flash("Email already registered", "danger")
        else:
            hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username=username, email=email, password=hashed_pw)
            db.session.add(user)
            db.session.commit()
            flash("Registered successfully! Please log in.", "success")
            return redirect(url_for('login'))
    return render_template("register.html")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out", "info")
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/methodology')
def methodology():
    return render_template("methodology.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True,host='0.0.0.0', port=8000)
