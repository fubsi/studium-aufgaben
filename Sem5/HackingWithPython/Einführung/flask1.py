from flask import Flask

app = Flask(__name__)

@app.route('/')
def home_page():
    return "home page"

@app.route('/home')
def hello_world():
    return "<h1>Hello World!</h1><p>this home</p>"

if __name__ == "__main__":
    app.run(debug=True)