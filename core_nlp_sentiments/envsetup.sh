#!/bin/bash

# User specific aliases and functions
# DO NOT change the classpath order!!!

export CLASSPATH="$CLASSPATH:.:./bin/:./config/"
for file in `find /home/mf3jh/workspace/lib/stanford-corenlp-4.2.2/  -name "*.jar"`; do export CLASSPATH="$CLASSPATH:`realpath $file`"; done
export _JAVA_OPTIONS="-Djava.net.preferIPv4Stack=true"
