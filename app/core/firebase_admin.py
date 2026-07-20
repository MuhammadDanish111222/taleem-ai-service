import firebase_admin
from firebase_admin import credentials, auth
from app.core.config import get_settings

_firebase_app = None

def get_firebase_app():
    global _firebase_app
    if _firebase_app is None:
        settings = get_settings()
        if not settings.FIREBASE_ADMIN_PROJECT_ID:
            raise ValueError("Firebase Admin SDK not configured.")
        
        # Replace unescaped newlines in private key
        private_key = settings.FIREBASE_ADMIN_PRIVATE_KEY.replace("\\n", "\n")
        
        cred = credentials.Certificate({
            "type": "service_account",
            "project_id": settings.FIREBASE_ADMIN_PROJECT_ID,
            "private_key": private_key,
            "client_email": settings.FIREBASE_ADMIN_CLIENT_EMAIL,
            "token_uri": "https://oauth2.googleapis.com/token",
        })
        _firebase_app = firebase_admin.initialize_app(cred)
    return _firebase_app

def get_auth():
    get_firebase_app()
    return auth
