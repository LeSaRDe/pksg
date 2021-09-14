#!/bin/bash

cd %HOME%/core_nlp_sentiments/
. envsetup_uva.sh
make
echo "CoreNLP TXT SENT $2 Starts..."
java -Xmx300G core_nlp_sentiments.PhraseSentimentParallel $1 $2
cd %HOME%
echo "CoreNLP TXT SENT $2 Done."
