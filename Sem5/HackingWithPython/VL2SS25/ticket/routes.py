from ticket import app, db
from flask import render_template, request, redirect, url_for
from sqlalchemy import text

@app.route('/')
def home_page():
    return render_template("home.html")

@app.route('/tickets')
def ticket_page():
    #items = [{"id": 1, "priority": 2, "username": "Marc", "title": "something broke"},
    #         {"id": 2, "priority": 1, "username": "Barc", "title": "nothing worky"},
    #         {"id": 3, "priority": 3, "username": "Quarc", "title": "idk man.."}]
    query_stmt = f"SELECT * FROM ticket"
    result = db.session.execute(text(query_stmt))
    items = result.fetchall()

    print(items)


    return render_template("tickets.html", items=items)

@app.route('/login', methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        print("POST request received")
        username = request.form.get("Username")
        password = request.form.get("Password")
        print(f"Username: {username}, Password: {password}")

        query_stmt = f"SELECT * FROM benutzer WHERE benutzername = '{username}' AND passwort = '{password}'"
        result = db.session.execute(text(query_stmt))
        items = result.fetchall()
        print(items)

        if len(items) > 0:
            print("User exists")
            resp = redirect(url_for('ticket_page'))
            resp.set_cookie("benutzerId", str(items[0][0]))
            return resp

    return render_template("login.html")

@app.route('/register', methods=["GET", "POST"])
def register_page():
    if request.method == "POST":
        print("POST request received")
        username = request.form.get("Username")
        password = request.form.get("Password")
        print(f"Username: {username}, Password: {password}")
    return render_template("register.html")