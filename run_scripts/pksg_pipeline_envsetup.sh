#! /bin/bash

##################################################
# ENVIRONMENT VARIABLES
##################################################
echo "WORKSPACE $WORKSPACE"
echo "OUTPUT $OUTPUT"

if [ ! -d "$OUTPUT" ]
then
    echo "Creating output dir ${OUTPUT}"
    mkdir $OUTPUT
    echo "Output dir created"
else
    echo "Output dir exists"
fi