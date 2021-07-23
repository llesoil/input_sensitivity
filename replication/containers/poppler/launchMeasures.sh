#!/bin/bash

set -xv

mkdir 'data'
# for each input (here a pdf file)
cat listInputs.csv | while read line
do
  read -d, inputname < <(echo $line)
  path="./inputs/$inputname.pdf"
  # list the runtime configs
  configs=`ls ./scripts/*.sh`
  # a csv per input and per compile-time option
  csvOutput="./data/$inputname.csv"
  # do not create if the file already exists
  if test -f "$csvOutput"; then
    echo "$csvOutput already exists"
  else
    echo "Starting to work with input: $inputname"
    # columns names
    header="configurationID,format,j,jp2,jbig2,ccitt,size,realtime"
    touch $csvOutput
    cat /dev/null > $csvOutput
    echo "$header" > $csvOutput
    # we execute all the runtime configs
    for config in $configs
    do
      echo "Processing: $config"
      csvLine=`bash $config $path`
      # and store them in the csv
      echo "$csvLine" >> $csvOutput
    done
fi
done

echo "Done with input $inputname"
