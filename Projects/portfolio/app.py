from flask import Flask, render_template, request, url_for, redirect

app = Flask(__name__)


# @app.route('/')
# def NavBar():
#     return render_template('navbar.html')


@app.route('/')
def home():
    return render_template('navbar.html')


if __name__ == '__main__':
    app.run(debug=True)
