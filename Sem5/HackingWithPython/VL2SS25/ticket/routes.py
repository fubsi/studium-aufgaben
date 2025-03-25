from ticket import app
from flask import render_template, request

@app.route('/')
def home_page():
    return render_template("home.html")

@app.route('/tickets')
def ticket_page():
    items = [{"id": 1, "priority": 2, "username": "Marc", "title": "something broke"},
             {"id": 2, "priority": 1, "username": "Barc", "title": "nothing worky"},
             {"id": 3, "priority": 3, "username": "Quarc", "title": "idk man.."}]
    
    return render_template("tickets.html", items=items)

@app.route('/login', methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        print("POST request received")
        username = request.form.get("Username")
        password = request.form.get("Password")
        print(f"Username: {username}, Password: {password}")
    return render_template("login.html")

@app.route('/register', methods=["GET", "POST"])
def register_page():
    if request.method == "POST":
        print("POST request received")
        username = request.form.get("Username")
        password = request.form.get("Password")
        print(f"Username: {username}, Password: {password}")
    return render_template("register.html")