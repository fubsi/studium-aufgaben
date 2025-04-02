from library import app, db
from flask import render_template, request, redirect, url_for
from sqlalchemy import text

@app.route('/')
def home():
    benutzername = get_benutzername(request)
    return render_template('home.html', benutzername=benutzername)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        benutzername = request.form['benutzername']
        password = request.form['password']
        # Here you would typically check the benutzername and password against a database
        print(f"Recieved login request for {benutzername} with password {password}")

        query = f"SELECT * FROM benutzer WHERE benutzername='{benutzername}' AND password='{password}'"
        result = db.session.execute(text(query)).fetchall()
        if len(result) > 0:
            # Assuming the user is authenticated successfully
            resp = redirect(url_for('home'))
            resp.set_cookie('benutzername', result[0][1])  # Set a cookie with the benutzername
            return resp

        
        return redirect(url_for('home'))
    
    # If the request method is GET, just render the login page
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        benutzername = request.form['benutzername']
        password = request.form['password']
        # Here you would typically save the new user to a database
        print(f"Recieved registration request for {benutzername} with password {password}")

        query = f"SELECT * FROM benutzer WHERE benutzername='{benutzername}'"
        result = db.session.execute(text(query)).fetchall()

        if len(result) == 0:
            # Assuming the user is registered successfully
            insert_query = f"INSERT INTO benutzer (benutzername, passwort) VALUES ('{benutzername}', '{password}')"
            db.session.execute(text(insert_query))
            db.session.commit()
            return redirect(url_for('login'))
        
        redirect(url_for('register'))
    
    # If the request method is GET, just render the register page
    return render_template('register.html')

@app.route('/logout')
def logout():
    resp = redirect(url_for('home'))
    resp.set_cookie('benutzername', '', expires=0)  # Clear the cookie
    return resp

@app.route('/library')
def library():
    benutzername = get_benutzername(request)
    
    query = f"SELECT * FROM Buch WHERE benutzerid = (SELECT id FROM benutzer WHERE benutzername = {benutzername})"
    result = db.session.execute(text(query)).fetchall()

    if len(result) == 0:
        print(f"No books found for user {benutzername}")
        return render_template('library.html', benutzername=benutzername, books=[])
    
    print(f"Recieved request for library page with benutzername {benutzername} and books {result}")
    return render_template('library.html', benutzername=benutzername, books=result)

def get_benutzername(request):
    benutzername = request.cookies.get('benutzername')
    print(f"Recieved request for library page with benutzername {benutzername}")
    if benutzername is None:
        return 'Guest'
    return benutzername