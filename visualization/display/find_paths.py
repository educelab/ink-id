import os
import json
import logging
import glob

# Logging config
logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.ERROR)

#####  StringFilter Function #####

def stringfilter(candidate, filter_string):
    return True if filter_string in candidate else False

def get_viewer_sets(datasets, groups, columns, truths,samples, model, root_dir='.'):
    '''
    For the viewer page. 
    '''
    substrings =[]

    for dataset in datasets:
        for group in groups:
            for column in columns:
                for truth in truths:
                    for sample in samples:
                        substrings.append(
                            'd'+dataset+'-g'+group+'-c'+column[-1]+'-t'+truth+'-s'+sample)
    
    logging.debug("substrings: %s", substrings)

    subvolume_paths =[]

    for substring in substrings:
        subvolume={}
        prefix = root_dir + '/**/'+substring

        subvolume['name']=substring
        subvolume['plotlymonoImg']=glob.glob(prefix+'-iPlotlymono.png', recursive=True)[0]
        subvolume['plotlymonoMeta']=glob.glob(prefix+'-iPlotlymono.json', recursive=True)[0]
        subvolume['plotlycolorImg']=glob.glob(prefix+'-iPlotlycolor.png', recursive=True)[0]
        subvolume['plotlycolorMeta']=glob.glob(prefix+'-iPlotlycolor.json', recursive=True)[0]
        subvolume['ytcolorImg']=glob.glob(prefix+'-iYtcolor.png', recursive=True)[0]
        subvolume['ytcolorMeta']=glob.glob(prefix+'-iYtcolor.json', recursive=True)[0]
        subvolume['gradcamDefaultImg']=glob.glob(prefix+'-m'+model+'*-iGradcamDefault.png', recursive=True)[0]
        subvolume['gradcamDefaultMeta']=glob.glob(prefix+'-m'+model+'*-iGradcamDefault.json', recursive=True)[0]
        subvolume['gradcamReverseImg']=glob.glob(prefix+'-m'+model+'*-iGradcamReverse.png', recursive=True)[0]
        subvolume['gradcamReverseMeta']=glob.glob(prefix+'-m'+model+'*-iGradcamReverse.json', recursive=True)[0]
        
        subvolume_paths.append(subvolume)

    logging.debug("final subvolume_paths: %s", subvolume_paths)

    return(dict(subvolumes=subvolume_paths))



def retrieve_images(#root_dir='/home/mhaya2/3d-utilities/SubvolumeVisualization/Results',
                    root_dir=".",
                    dataset_sel=None, group_sel=None, col_sel=None, 
                    truth_sel=None, prediction_sel=None, model_sel=None, 
                    image_sel=None):

    '''
    dataset_sel: "CarbonPhantom"
    group_sel: "Interpolated" "NearestNeighbor"
    col_sel: "Col2" "Col6"
    truth_sel: "CarbonInk" "NoInk" "IronGall"
    model_sel: "1254552_360000" 
    prediction_sel = "ink" "no_ink"
    image_sel: "Plotlymono" "Plotlycolor" "Ytcolor" "GradcamDefault" "GradcamReverse" 
                "SimpleSubvolume", "Superimposed"  
    '''

    logging.debug("Input root_dir: %s", root_dir)
    logging.debug("Input dataset_sel: %s", dataset_sel)
    logging.debug("Input group_sel: %s", group_sel)
    logging.debug("Input col_sel: %s", col_sel)
    logging.debug("Input truth_sel: %s", truth_sel)
    logging.debug("Input prediction_sel: %s", prediction_sel)
    logging.debug("Input image_sel: %s", image_sel)
    
    # Find all the files in subdirectories recursively 
    f_list = []
    for path, dirs, files in os.walk(root_dir, followlinks=True):
        for file in files:
            f_list.append(os.path.join(path, file))
   
    # just focus on image files (ignore json with the same name)
    f_list = [x for x in f_list if 'png' in x]

    logging.debug("Found files: %s", f_list)
    logging.debug("Number of found files: %d", len(f_list))

    #### Dataset Filter ####
    
    if dataset_sel:
        dataset_filtered = list(filter(lambda x: stringfilter(x, 'd'+dataset_sel),
                                f_list))
    else:
        dataset_filtered = f_list
    
    logging.debug("dataset_filtered: %s", dataset_filtered)
    logging.debug("dataset_filtered size: %d", len(dataset_filtered))


    #### Group Filter ####
    if group_sel:
        group_filtered = list(filter(lambda x: stringfilter(x, 'g'+group_sel), dataset_filtered))
    else:
        group_filtered = dataset_filtered
    
    logging.debug("group_filtered: %s", group_filtered)
    logging.debug("group_filtered size: %d", len(group_filtered))


    #### Column Filter ####
    # Just strip the number
    if col_sel:
        col_filtered = list(filter(lambda x: stringfilter(x, 'c'+col_sel[-1]), group_filtered))
    else:
        col_filtered = group_filtered
    
    logging.debug("col_filtered: %s", col_filtered)
    logging.debug("col_filtered size: %d", len(col_filtered))


    #### Truth Filter ####
    if truth_sel:
        truth_filtered = list(filter(lambda x: stringfilter(x, 't'+truth_sel), col_filtered))
    else:
        truth_filtered = col_filtered
    
    logging.debug("truth_filtered: %s", truth_filtered)
    logging.debug("truth_filtered size: %d", len(truth_filtered))
    
    #### Model Filter ####
    if model_sel:
        model_filtered = list(filter(lambda x: stringfilter(x, 'm' +model_sel), truth_filtered))
    else:
        model_filtered = truth_filtered
    
    logging.debug("model_filtered: %s", model_filtered)
    logging.debug("model_filtered size: %d", len(model_filtered))
    
    #### Prediction Filter ####
    prediction_ink = []
    prediction_no_ink = []
    
    if prediction_sel:
        pred = '0' if prediction_sel=="no_ink" else '1'
        pred_filtered = list(filter(lambda x: stringfilter(x, 'p'+pred), model_filtered))
    else:
        pred_filtered = model_filtered
    
    logging.info("pred_filtered: %s", pred_filtered)
    logging.info("pred_filtered size: %d", len(pred_filtered))
    

    #### Image Filter ####
    if image_sel:
        image_filtered = list(filter(lambda x: stringfilter(x, 'i'+image_sel), pred_filtered))
    else:
        image_filtered = pred_filtered
    
    logging.debug("image_filtered: %s", image_filtered)
    logging.debug("image_filtered size: %d", len(image_filtered))
    
        
    paths = []   # list of dictionaries 
    #[{'image': 'path/to/image', 'metadata': 'path/to/metadata'}, ]

    for image_path in image_filtered:
        temp_d = {}
        temp_d['image'] = image_path
        temp_d['metadata'] = image_path[:-3] + 'json'
        paths.append(temp_d)

    
    logging.debug("final paths output: %s", paths)
    logging.info("final list (paths) length: %d", len(paths))
    
    
    ###########  Create a JSON with {'file_paths': [<str>], 'directory_paths': [<str>,]} ########
    
    d={}
    d['paths'] = paths
    
    logging.info("FINAL JSON: %s", json.dumps(d, indent=4))

    # return a simple dictionary
    return(d)

