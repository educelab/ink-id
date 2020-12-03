from flask import Flask, render_template, request, json, jsonify
import logging
import find_paths as fp

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

@app.route('/imagePaths', methods=['POST'])
def get_paths():

    json_request = request.json

    dataset_sel = json_request['dataset'] 
    group_sel = None if json_request['group'] == "All" else json_request['group']
    col_sel = None if json_request['column'] == "All" else json_request['column']
    truth_sel = None if json_request['truth'] == "All" else json_request['truth']
    prediction_sel = None if json_request['prediction'] == "All" else json_request['prediction']
    image_sel = None if json_request['imagetype'] == "All" else json_request['imagetype']

    logging.debug("Input values: %s, %s, %s, %s, %s, %s", dataset_sel, group_sel,
                  col_sel, truth_sel, prediction_sel, image_sel)

    
    return_value = fp.retrieve_images(root_dir='/home/mhaya2/3d-utilities/SubvolumeVisualization/Results/1203/',
                                      dataset_sel=dataset_sel, 
                                      group_sel=group_sel,
                                      col_sel=col_sel,
                                      truth_sel=truth_sel,
                                      prediction_sel=prediction_sel,
                                      image_sel=image_sel)

    logging.debug("Returned Json is (app.py): %s", return_value)


    return(return_value)
