"""
Make predictions using a trained model.
"""

import argparse

import tensorflow as tf

from inkid.volumes import VolumeSet
import inkid.model
import inkid.ops

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', metavar='path', required=True,
                        help='path to trained model directory')
    parser.add_argument('--data', metavar='path', required=True,
                        help='path to volume data (slices directory)')
    parser.add_argument('--groundtruth', metavar='path', required=True,
                        help='path to ground truth image')
    parser.add_argument('--surfacemask', metavar='path', required=True,
                        help='path to surface mask image')
    parser.add_argument('--surfacedata', metavar='path', required=True,
                        help='path to surface data')

    args = parser.parse_args()

    params = inkid.ops.load_default_parameters()

    # Adjust some parameters from supplied arguments
    params['volumes'][0]['data_path'] = args.data
    params['volumes'][0]['ground_truth'] = args.groundtruth
    params['volumes'][0]['surface_mask'] = args.surfacemask
    params['volumes'][0]['surface_data'] = args.surfacedata
    # params['output_path'] = os.path.join(
    #     args.outputdir,
    #     '3dcnn-predictions',
    #     datetime.datetime.today().strftime('%Y-%m-%d_%H.%M.%S')
    # )

    volumes = VolumeSet(params)

    # with tf.Session() as sess:
    #     inputs, labels = volumes.prediction_input_fn(30)

    #     inputs = sess.run((inputs))

    #     for (n, i) in enumerate(inputs['Subvolume']):
    #         inkid.ops.save_volume_to_image_stack(i, str(n))

    print(args.model)
    estimator = tf.estimator.Estimator(
        model_fn=inkid.model.model_fn_3dcnn,
        model_dir=args.model,
        params={
            'drop_rate': params['drop_rate'],
            'subvolume_shape': params['subvolume_shape'],
            'batch_norm_momentum': params['batch_norm_momentum'],
            'filters': params['filters'],
            'learning_rate': params['learning_rate'],
        },
    )

    print(estimator.get_variable_names())
    print(estimator.get_variable_value('global_step'))
    
    predictions = estimator.predict(
        input_fn=lambda: volumes.prediction_input_fn(
            params['prediction_batch_size'],
        ),
        predict_keys=['classes', 'probabilities'],
    )

    # print(list(predictions))
    
    

if __name__ == '__main__':
    main()
