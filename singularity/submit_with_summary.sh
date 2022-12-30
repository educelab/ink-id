#!/bin/bash

# This script submits sbatch jobs for both the training jobs and the subsequent
# summary and upload scripts.

# Run sbatch job as provided by just executing all args
# (which includes call to submit.sh as the first arg)
OUTPUT=$("$@")
# Should print "Submitted batch job <ID>"
echo $OUTPUT
# Get the job ID from that string
JOB_ID=$(echo $OUTPUT | sed 's/[^0-9]*//g')

# Need to get the output directory from the arguments
# Shift args until we reach "--output", meaning <outputDir> follows
while [ $1 != "--output" ]
do
  shift
done
# "$2" is now <outputDir> and can be passed to summary script

# Make a new summary job dependent on the first one
OUTPUT2=$(sbatch --dependency afterany:"$JOB_ID" summary.sh "$2")
# Should print "Submitted batch job <ID>"
echo $OUTPUT2
# Get the job ID from that string
JOB_ID2=$(echo $OUTPUT2 | sed 's/[^0-9]*//g')
