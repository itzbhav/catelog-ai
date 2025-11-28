from flask import Flask
import os

print("=== STARTING APP.PY ===")

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Railway!"

if __name__ == '__main__':
    print("=== ABOUT TO START FLASK ===")
    port = int(os.getenv("PORT", 8080))
    print("PORT =", port)
    app.run(host='0.0.0.0', port=port, debug=False)
