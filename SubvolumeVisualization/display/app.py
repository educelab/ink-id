from flask import Flask, render_template, request, json, jsonify, redirect, url_for, session, flash
from functools import wraps
from datetime import timedelta
import configparser
import logging
import find_paths as fp

# Logging 
logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.ERROR)

config = configparser.ConfigParser()
config_file = 'display.conf'
config.read(config_file)


app = Flask(__name__)

app.secret_key = config['AUTHENTICATION']['AppPassword']

@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=5)


# login required decorator
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('You need to login first.')
            return redirect(url_for('login'))
    return wrap

# Route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        saved_username = config['AUTHENTICATION']['username'] 
        saved_password = config['AUTHENTICATION']['password']

        logging.debug("saved username: %s", saved_username)
        logging.debug("saved password: %s", saved_password)
        logging.debug("username received: %s", request.form['username'])
        logging.debug("password received: %s", request.form['password'])

        if request.form['username'] !=  saved_username or request.form['password'] != saved_password:
            error = 'Invalid Credentials. Please try again.'
        else:
            session['logged_in'] = True
            return redirect(url_for('index'))
    return render_template('login.html', error=error)

@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html', title="Home")

@app.route('/viewer', methods=['GET', 'POST'])
@login_required
def viewer():
    if request.method == 'POST':

        json_request = request.json

        choices = {
            "datasets":["CarbonPhantom"],
	    "groups": ["Interpolated", "NearestNeighbor"],
	    "columns": ["Col2", "Col6"],
	    "truths": ["CarbonInk", "NoInk","IronGall"],
	    "samples": ["0", "1", "2", "3", "4", "5", "6", "7","8"]
        }    

        # Converting All to actual values
        dataset_sel = [d for d in choices["datasets"]] if json_request['dataset']=="All" \
                    else [json_request['dataset']] 
        group_sel = [g for g in choices["groups"]] if json_request['group'] == "All" \
                    else [json_request['group']]
        col_sel = [c for c in choices["columns"]] if json_request['column'] == "All" \
                    else [json_request['column']]
        truth_sel = [t for t in choices["truths"]] if json_request['truth'] == "All" \
                    else [json_request['truth']]
        sample_sel = [s for s in choices["samples"]] if json_request['sample'] == "All" \
                    else [json_request['sample']]
        model_sel = json_request['model']

        #logging.debug("dataset_sel: %s", dataset_sel)
        #logging.debug("group_sel: %s", group_sel)
        #logging.debug("col_sel: %s", col_sel)
        #logging.debug("truth_sel: %s", truth_sel)
        #logging.debug("sample_sel: %s", sample_sel)
        #logging.debug("model_sel: %s", model_sel)

        
        return_value = fp.get_viewer_sets(
                        datasets=dataset_sel,
                        groups=group_sel,
                        columns=col_sel,
                        truths=truth_sel,
                        samples=sample_sel,
                        model=model_sel,
                        root_dir='static/images/results'
                        )

        # add '/' to each path
        for subvolume in return_value['subvolumes']:
            for key in subvolume:
                if key != "name":
                    subvolume[key] = '/'+subvolume[key]

        logging.debug("Modified result is: %s", json.dumps(return_value))


        return(json.dumps(return_value)) 


    return render_template('viewer.html', title="Viewer")

@app.route('/filter', methods=['GET', 'POST'])
@login_required
def filter():
    if request.method == 'POST':
        json_request = request.json

        dataset_sel = json_request['dataset'] 
        group_sel = None if json_request['group'] == "All" else json_request['group']
        col_sel = None if json_request['column'] == "All" else json_request['column']
        truth_sel = None if json_request['truth'] == "All" else json_request['truth']
        prediction_sel = None if json_request['prediction'] == "All" else json_request['prediction']
        image_sel = None if json_request['imagetype'] == "All" else json_request['imagetype']
        model_sel = None if json_request['model'] == "All" else json_request['model']

        logging.debug("Input values: %s, %s, %s, %s, %s, %s, %s", 
                      dataset_sel, group_sel, col_sel, truth_sel, prediction_sel, 
                      image_sel, model_sel)

        
        return_value = fp.retrieve_images(root_dir='static/images/results',
                                          dataset_sel=dataset_sel, 
                                          group_sel=group_sel,
                                          col_sel=col_sel,
                                          truth_sel=truth_sel,
                                          prediction_sel=prediction_sel,
                                          image_sel=image_sel,
                                          model_sel=model_sel)

        # add '/' to each path
        for path in return_value['paths']:
            path['image'] = '/' + path['image']
            path['metadata'] = '/' +path['metadata']

        logging.debug("Returned Json is (app.py): %s", json.dumps(return_value))


        return(json.dumps(return_value))
        

    return render_template('filter.html', title="Gallery")
  

