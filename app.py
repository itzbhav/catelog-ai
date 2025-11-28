from flask import Flask
from flask_session import Session
import os

print("Imported Flask and Flask-Session")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'test-secret')
app.config['SESSION_TYPE'] = 'null'   # Try 'null' for cloud stateless, not 'filesystem'
Session(app)

@app.route('/')
def home():
    return "App started successfully!"

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    print("PORT =", port)
    app.run(host='0.0.0.0', port=port, debug=False)
