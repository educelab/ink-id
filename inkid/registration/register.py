import argparse
import pathlib

import itk
from matplotlib import pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-image", type=pathlib.Path, required=True)
    parser.add_argument("--moving-image", type=pathlib.Path, required=True)
    args = parser.parse_args()

    pixel_type = itk.F

    print("Reading images...")
    fixed_img = itk.imread(str(args.fixed_image), pixel_type)
    moving_img = itk.imread(str(args.moving_image), pixel_type)

    print("Downsampling images...")
    scale_down_factor = 4
    output_size = fixed_img.GetLargestPossibleRegion().GetSize()
    output_size[0] //= scale_down_factor
    output_size[1] //= scale_down_factor
    output_spacing = fixed_img.GetSpacing()
    output_spacing[0] = fixed_img.GetSpacing()[0] * scale_down_factor
    output_spacing[1] = fixed_img.GetSpacing()[1] * scale_down_factor
    fixed_img = itk.resample_image_filter(
        fixed_img,
        size=output_size,
        output_spacing=output_spacing,
    )
    moving_img = itk.resample_image_filter(
        moving_img,
        size=output_size,
        output_spacing=output_spacing,
    )

    print("Setting up registration pipeline...")
    initial_difference = itk.SubtractImageFilter.New(Input1=fixed_img, Input2=moving_img)
    initial_difference_img = itk.output(initial_difference)

    dimension = fixed_img.GetImageDimension()
    image_type = itk.Image[pixel_type, dimension]
    spline_order = 3

    transform_type = itk.BSplineTransform[itk.D, dimension, spline_order]
    # transform_type = itk.TranslationTransform[itk.D, dimension]
    initial_transform = transform_type.New()

    # optimizer = itk.RegularStepGradientDescentOptimizerv4.New(
    #     LearningRate=4,
    #     MinimumStepLength=0.001,
    #     RelaxationFactor=0.5,
    #     NumberOfIterations=200,
    #     GradientMagnitudeTolerance=1e-7,
    # )
    optimizer = itk.LBFGSOptimizer.New(
        # CostFunctionConvergenceFactor=1e+12,
        # GradientConvergenceTolerance=1.0e-35,
        # NumberOfIterations=5,
        # MaximumNumberOfFunctionEvaluations=500,
        # MaximumNumberOfCorrections=5,
    )
    # print(optimizer.GetBoundSelection())


    metric = itk.MattesMutualInformationImageToImageMetricv4[image_type, image_type].New()

    registration = itk.ImageRegistrationMethodv4.New(
        FixedImage=fixed_img,
        MovingImage=moving_img,
        Metric=metric,
        Optimizer=optimizer,
        InitialTransform=initial_transform,
    )

    identity_transform = transform_type.New()
    identity_transform.SetIdentity()
    registration.SetMovingInitialTransform(identity_transform)
    registration.SetFixedInitialTransform(identity_transform)

    registration.SetNumberOfLevels(1)
    registration.SetSmoothingSigmasPerLevel([0])
    registration.SetShrinkFactorsPerLevel([1])

    print("Performing registration...")
    registration.Update()

    number_of_iterations = optimizer.GetCurrentIteration()
    best_metric_value = optimizer.GetValue()
    print(f"Iterations: {number_of_iterations}, metric: {best_metric_value}")
    print(f"Stop condition: {registration.GetOptimizer().GetStopConditionDescription()}")

    output_transform = registration.GetTransform()
    resampler = itk.ResampleImageFilter.New(
        Input=moving_img,
        Transform=output_transform,
        UseReferenceImage=True,
        ReferenceImage=fixed_img,
    )
    resampler.SetDefaultPixelValue(0)
    resampled_img = itk.output(resampler)

    new_difference = itk.SubtractImageFilter.New(Input1=fixed_img, Input2=resampled_img)
    new_difference_img = itk.output(new_difference)

    moving_difference = itk.SubtractImageFilter.New(Input1=moving_img, Input2=resampled_img)
    moving_difference_img = itk.output(moving_difference)

    composite_img = np.hstack(
        [
            itk.array_from_image(fixed_img),
            itk.array_from_image(initial_difference_img),
            itk.array_from_image(new_difference_img),
            itk.array_from_image(moving_difference_img),
        ],
    )

    plt.imshow(composite_img)
    plt.show()


if __name__ == "__main__":
    main()
