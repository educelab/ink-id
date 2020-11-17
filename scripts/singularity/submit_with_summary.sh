#!/bin/bash

# Run sbatch job as provided by just executing all args
# (which includes call to submit.sh as the first arg)
OUTPUT=$("$@")
# Should print "Submitted batch job <ID>"
echo $OUTPUT
# Get the job ID from that string
JOB_ID=$(echo $OUTPUT | sed 's/[^0-9]*//g')

# Need to get the output directory from the arguments
# Shift args until we reach "submit.sh", meaning <infile> <outputDir> follow
while [ $1 != "submit.sh" ]
do
  shift
done
# "$3" is now <outputDir> and can be passed to summary script

# Make a new summary job dependent on the first one
OUTPUT2=$(sbatch --dependency afterany:"$JOB_ID" summary.sh "$3")
# Should print "Submitted batch job <ID>"
echo $OUTPUT2
# Get the job ID from that string
JOB_ID2=$(echo $OUTPUT2 | sed 's/[^0-9]*//g')

# Make an rclone upload job dependent on the summary job
OUTPUT3=$(sbatch --dependency afterany:"$JOB_ID2" rclone_upload.sh "$3")
# Should print "Submitted batch job <ID>"
echo $OUTPUT3
