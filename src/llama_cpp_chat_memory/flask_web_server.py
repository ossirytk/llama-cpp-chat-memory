from flask import Flask, send_from_directory

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/test/")
def test():
    return "<p>TEST!</p>"


@app.route("/static/<path:path>")
def send_style(path):
    return send_from_directory("static", path)
