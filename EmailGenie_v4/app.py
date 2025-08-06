import os
import logging
import json
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    'pool_pre_ping': True,
    "pool_recycle": 300,
}

# Initialize extensions
db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Initialize SocketIO for real-time features
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Register custom Jinja2 filters
@app.template_filter('tojsonfilter')
def tojson_filter(obj):
    """Convert Python object to JSON string for use in templates"""
    return json.dumps(obj)

# Create tables
with app.app_context():
    import models  # noqa: F401
    db.create_all()
    logging.info("Database tables created")
