# app.py
import os
import logging
from datetime import timedelta
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from flask_jwt_extended import (
    JWTManager, create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity, verify_jwt_in_request, get_jwt
)
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------------- JWT Configuration ----------------
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-change-in-production-123456789')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)
app.config['JWT_TOKEN_LOCATION'] = ['cookies']
app.config['JWT_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['JWT_COOKIE_CSRF_PROTECT'] = False  # Enable in production
app.config['JWT_ACCESS_COOKIE_PATH'] = '/'
app.config['JWT_REFRESH_COOKIE_PATH'] = '/token/refresh'

jwt = JWTManager(app)

# ---------------- In-memory user database (Replace with real database in production) ----------------
USERS_DB = {
    'admin': {
        'password': generate_password_hash('admin123'),
        'email': 'admin@example.com',
        'role': 'admin'
    },
    'demo': {
        'password': generate_password_hash('demo123'),
        'email': 'demo@example.com',
        'role': 'user'
    },
    # Add your own users below:
    'abhinav': {
        'password': generate_password_hash('abhinav2024'),
        'email': 'abhinav@example.com',
        'role': 'user'
    },
    'john': {
        'password': generate_password_hash('john123'),
        'email': 'john@company.com',
        'role': 'user'
    },
    'sarah': {
        'password': generate_password_hash('sarah456'),
        'email': 'sarah@example.com',
        'role': 'admin'
    }
}

# ---------------- Authentication helpers ----------------
def login_required_page(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            verify_jwt_in_request()
            return f(*args, **kwargs)
        except Exception as e:
            return redirect(url_for('login'))
    return decorated_function


# ---------------- Model loading ----------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

logger.info(f"Loading tokenizer & model: {MODEL_NAME} (this may take a moment)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
logger.info("Model loaded.")


# ---------------- Academic text check ----------------
def is_academic_text(text: str) -> bool:
    academic_keywords = [
        "study", "research", "paper", "proposes", "investigates", "analysis",
        "experiment", "methodology", "results", "objective", "findings",
        "model", "dataset", "approach", "algorithm", "performance",
        "evaluation", "conclusion", "abstract", "method"
    ]
    if not text:
        return False
    text_lower = text.lower()
    if len(text.split()) < 20:
        return False
    return any(word in text_lower for word in academic_keywords)


# ---------------- Field detection ----------------
def detect_academic_field(text: str) -> str:
    text = text.lower()
    fields = {
        "cs.LG": ["machine learning", "deep learning", "neural", "model", "training", "dataset"],
        "cs.AI": ["artificial intelligence", "agent", "reasoning", "knowledge"],
        "cs.CV": ["image", "vision", "detection", "segmentation"],
        "cs.CL": ["nlp", "language", "bert", "text", "transformer"],
        "eess.SP": ["signal", "frequency", "audio", "speech"],
        "stat.ML": ["regression", "bayesian", "probability", "statistical"],
        "q-bio": ["genomics", "protein", "biological", "dna"],
        "physics.comp-ph": ["simulation", "quantum", "particle", "computational"],
        "econ.GN": ["economics", "finance", "market", "forecast"]
    }

    for field, keywords in fields.items():
        if any(k in text for k in keywords):
            return field
    return "cs.Other"


def get_field_name(field_code: str) -> str:
    names = {
        "cs.LG": "Machine Learning",
        "cs.AI": "Artificial Intelligence",
        "cs.CL": "Natural Language Processing",
        "cs.CV": "Computer Vision",
        "eess.SP": "Signal Processing",
        "stat.ML": "Statistical Machine Learning",
        "q-bio": "Quantitative Biology",
        "physics.comp-ph": "Computational Physics",
        "econ.GN": "Economics",
        "cs.Other": "Other Computer Science"
    }
    return names.get(field_code, "Unknown Field")


# ---------------- Sentiment classification ----------------
def classify_abstract(text: str):
    if not text.strip():
        return None
    return classifier(text, truncation=True, max_length=512)[0]


# ---------------- Sample abstracts ----------------
SAMPLE_ABSTRACTS = {
    "Select a sample...": "",
    "Renewable Energy & Power Grids": (
        "This paper investigates the integration of renewable energy sources "
        "into existing power grids, focusing on optimizing energy distribution "
        "and minimizing losses. We propose a novel algorithm for load balancing "
        "that improves grid stability and efficiency."
    ),
    "Deep Learning & Computer Vision": (
        "We present a novel deep learning architecture for image segmentation "
        "that improves accuracy and inference speed. Experiments on public "
        "benchmarks demonstrate state-of-the-art performance."
    ),
    "Natural Language Processing": (
        "This work introduces a new approach to sentiment analysis using pretrained "
        "transformers and contrastive learning, achieving improved robustness across domains."
    ),
    "Medical Research": (
        "Our study examines the efficacy of a new therapeutic approach in treating "
        "a chronic disease. We conducted a randomized controlled trial to evaluate outcomes."
    ),
    "Climate Science": (
        "This research analyzes the impact of climate change on regional precipitation "
        "patterns using long-term datasets and climate simulations."
    )
}


# ---------------- Routes ----------------
@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html")


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json()
    username = data.get('username', '')
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    user = USERS_DB.get(username)
    if not user or not check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid username or password'}), 401

    # Create tokens
    access_token = create_access_token(
        identity=username,
        additional_claims={'role': user['role'], 'email': user['email']}
    )
    refresh_token = create_refresh_token(identity=username)

    # Create response with cookies
    response = jsonify({
        'message': 'Login successful',
        'user': {
            'username': username,
            'email': user['email'],
            'role': user['role']
        }
    })
    
    # Set JWT cookies
    response.set_cookie('access_token_cookie', access_token, 
                       httponly=True, samesite='Lax', max_age=3600)
    response.set_cookie('refresh_token_cookie', refresh_token,
                       httponly=True, samesite='Lax', max_age=2592000)
    
    return response, 200


@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    email = data.get('email', '').strip()

    if not username or not password or not email:
        return jsonify({'error': 'All fields are required'}), 400

    if username in USERS_DB:
        return jsonify({'error': 'Username already exists'}), 409

    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    # Add new user
    USERS_DB[username] = {
        'password': generate_password_hash(password),
        'email': email,
        'role': 'user'
    }

    return jsonify({'message': 'Registration successful'}), 201


@app.route("/api/logout", methods=["POST"])
def api_logout():
    response = jsonify({'message': 'Logout successful'})
    response.set_cookie('access_token_cookie', '', expires=0)
    response.set_cookie('refresh_token_cookie', '', expires=0)
    return response, 200


@app.route("/api/user", methods=["GET"])
@jwt_required()
def get_user():
    current_user = get_jwt_identity()
    claims = get_jwt()
    
    return jsonify({
        'username': current_user,
        'email': claims.get('email'),
        'role': claims.get('role')
    }), 200


@app.route("/", methods=["GET"])
@login_required_page
def index():
    current_user = get_jwt_identity()
    return render_template("index.html",
                           abstract="",
                           label=None,
                           confidence=None,
                           field_code=None,
                           field_name=None,
                           sample_options=SAMPLE_ABSTRACTS,
                           username=current_user)


@app.route("/predict", methods=["POST"])
@login_required_page
def predict():
    current_user = get_jwt_identity()
    abstract = request.form.get("abstract", "")
    selected_sample = request.form.get("sample_select", "")

    if selected_sample and selected_sample != "Select a sample...":
        abstract = SAMPLE_ABSTRACTS[selected_sample]

    if not abstract.strip():
        return render_template("index.html",
                               warning="⚠️ Please enter an abstract.",
                               abstract="",
                               sample_options=SAMPLE_ABSTRACTS,
                               username=current_user)

    if not is_academic_text(abstract):
        return render_template("index.html",
                               warning="⚠️ This does NOT look like an academic abstract.",
                               abstract=abstract,
                               sample_options=SAMPLE_ABSTRACTS,
                               username=current_user)

    field_code = detect_academic_field(abstract)
    field_name = get_field_name(field_code)

    result = classify_abstract(abstract)
    label = result["label"]
    confidence = round(result["score"] * 100, 2)

    return render_template("index.html",
                           abstract=abstract,
                           label=label,
                           confidence=confidence,
                           field_code=field_code,
                           field_name=field_name,
                           sample_options=SAMPLE_ABSTRACTS,
                           username=current_user)


# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
