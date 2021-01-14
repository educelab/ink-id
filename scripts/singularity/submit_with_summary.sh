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

# Make a new summary job dependent on the first one
# Shift args until we reach "submit.sh", meaning <infile> <outputDir> follow
while [ $1 != "submit.sh" ]
do
  shift
done
# "$3" is now <outputDir> and can be passed to summary script
OUTPUT2=$(sbatch --dependency afterany:"$JOB_ID" summary.sh "$3")
# Should print "Submitted batch job <ID>"
echo $OUTPUT2
