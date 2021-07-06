from flask import Flask, render_template, redirect, url_for, request
import function as function
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def bmi():
    name = ""
    bmi = ''
    if request.method == "POST" and "username" in request.form:
        name = request.form.get("username")
        weight = float(request.form.get("weight"))
        height = float(request.form.get("height"))
        bmi = function.calc_bmi(weight, height)
    return render_template("bmi.html", name=name, bmi=bmi)


if __name__ == "__main__":
    app.run()
