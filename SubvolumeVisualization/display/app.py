from flask import Flask, render_template, request, json, jsonify
import logging

# Logging config
logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.ERROR)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', title="Home")

@app.route('/overview')
def overview():
    return render_template('overview.html', title="Overview")

@app.route('/filter')
def filter():
    return render_template('filter.html', title="Filter")
