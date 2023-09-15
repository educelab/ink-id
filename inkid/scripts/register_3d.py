import itk
import os
import argparse
import logging

def generate_file_list(directory):
    """Generate a list of .tif files from a directory."""
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".tif")])

def register_volumes(fixed_dir, moving_dir, output_dir):
    logging.info("Generating file lists for fixed and moving images...")
    fixed_image_files = generate_file_list(fixed_dir)
    moving_image_files = generate_file_list(moving_dir)

    logging.info("Reading the fixed and moving images...")
    DiskImageType = itk.Image[itk.US, 3]
    ImageType = itk.Image[itk.F, 3]
    fixed_image_reader = itk.ImageSeriesReader.New(FileNames=fixed_image_files)
    fixed_image_cast = itk.CastImageFilter[DiskImageType, ImageType].New(Input=fixed_image_reader.GetOutput())
    moving_image_reader = itk.ImageSeriesReader.New(FileNames=moving_image_files)
    moving_image_cast = itk.CastImageFilter[DiskImageType, ImageType].New(Input=moving_image_reader.GetOutput())

    logging.info("Setting up the registration components...")
    TransformType = itk.AffineTransform[itk.D, 3]
    MetricType = itk.MattesMutualInformationImageToImageMetric[ImageType, ImageType]
    OptimizerType = itk.RegularStepGradientDescentOptimizer
    RegistrationType = itk.MultiResolutionImageRegistrationMethod[ImageType, ImageType]

    transform = TransformType.New()

    metric = MetricType.New()
    metric.SetNumberOfHistogramBins(50)
    
    optimizer = OptimizerType.New()
    optimizer.SetMinimumStepLength(0.0001)
    optimizer.SetNumberOfIterations(200)
    optimizer.SetRelaxationFactor(0.5)

    logging.info("Setting up the multi-resolution pyramid...")
    ImagePyramidType = itk.MultiResolutionPyramidImageFilter[ImageType, ImageType]
    fixed_image_pyramid = ImagePyramidType.New()
    moving_image_pyramid = ImagePyramidType.New()

    num_levels = 3
    fixed_image_pyramid.SetNumberOfLevels(num_levels)
    moving_image_pyramid.SetNumberOfLevels(num_levels)

    registration = RegistrationType.New()
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetTransform(transform)
    registration.SetInitialTransformParameters(transform.GetParameters())

    registration.SetNumberOfLevels(num_levels)
    registration.SetFixedImagePyramid(fixed_image_pyramid)
    registration.SetMovingImagePyramid(moving_image_pyramid)

    registration.SetFixedImage(fixed_image_cast.GetOutput())
    registration.SetMovingImage(moving_image_cast.GetOutput())

    # fixed_image_reader.Update()
    # registration.SetFixedImageRegion(fixed_image_reader.GetOutput().GetBufferedRegion())

    logging.info("Performing the registration...")
    registration.Update()

    logging.info("Applying the transform to the moving image...")
    resampler = itk.ResampleImageFilter.New(Input=moving_image_reader.GetOutput(),
                                            Transform=registration.GetTransform(),
                                            UseReferenceImage=True,
                                            ReferenceImage=fixed_image_reader.GetOutput())
    resampled_image = resampler.GetOutput()

    logging.info(f"Checking if output directory ({output_dir}) exists or needs creation...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info("Saving the registered volume to a directory of .tif files...")
    names_generator = itk.NumericSeriesFileNames.New()
    names_generator.SetStartIndex(0)
    names_generator.SetEndIndex(resampled_image.GetLargestPossibleRegion().GetSize()[2] - 1)  # Assuming z is the slice axis
    names_generator.SetIncrementIndex(1)
    names_generator.SetSeriesFormat(os.path.join(output_dir, "slice%03d.tif"))

    writer = itk.ImageSeriesWriter.New(Input=resampled_image, FileNames=names_generator.GetFileNames())
    writer.Update()

    logging.info("Registration and saving process completed successfully.")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="Register two 3D volumes from directories of .tif files.")
    parser.add_argument("--fixed_dir", required=True, help="Directory containing .tif files for the fixed image.")
    parser.add_argument("--moving_dir", required=True, help="Directory containing .tif files for the moving image.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the registered volume as .tif files.")
    
    args = parser.parse_args()
    register_volumes(args.fixed_dir, args.moving_dir, args.output_dir)
