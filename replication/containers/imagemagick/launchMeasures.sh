#!/bin/bash

set -xv

cat listInputs.csv | while read line
do
  read -d, inputname < <(echo $line)
  configs=`ls ./scripts/*.sh`
  csvOutput="./data/$inputname.csv"
  if test -f "$csvOutput"; then
    echo "$csvOutput already exists"
  else
    echo "Starting to work with input: $inputname"
    # columns names
    header="configurationID,memory,posterize,gaussianblur,thread,quality,size,t,time"
    touch $csvOutput
    cat /dev/null > $csvOutput
    echo "$header" > $csvOutput
    # we execute all the runtime configs
    for config in $configs
    do
      echo "Processing: $config"
      csvLine=`bash $config $inputname`
      # and store them in the csv
      echo "$csvLine" >> $csvOutput
    done
fi
done

echo "Done with input $inputname"
