from flask import Flask, flash, render_template, url_for, redirect, session, request
from datetime import timedelta

app = Flask(__name__)
app.secret_key = 'helololo'
app.permanent_session_lifetime = timedelta(minutes=1)


@app.route('/')
def home():
    return render_template('home.html')


@app.route("/projects")
def projects():
    return render_template('projects.html')


@app.route("/learn_with_me")
def learn_with_me():
    return render_template('learn_with_me.html')


@app.route("/about_me")
def about_me():
    return render_template('aboutme.html')


@app.route("/contact")
def contact():
    return render_template('contact.html')


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session.permanent = True
        user = request.form['username']
        session['user'] = user
        flash("Login Succesful!")
        return redirect(url_for("user"))
    else:
        if "user" in session:
            flash("Already logged in!")
            return redirect(url_for('user'))
        return render_template("login.html")


@app.route("/user")
def user():
    if 'user' in session:
        user = session['user']
        return render_template('user.html', user=user)
    else:
        return redirect(url_for('login'))


@app.route("/logout")
def logout():
    flash("You have been  logout", category="info")
    session.pop('user', None)
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)
