#!/bin/bash

set -xv

mkdir 'data'
# for each input (here a sat formula)
cat listInputs.csv | while read line
do
  read -d, inputname < <(echo $line)
  path="./inputs/$inputname.cnf"
  # list the runtime configs
  configs=`ls ./scripts/*.sh`
  # a csv per input
  csvOutput="./data/$inputname.csv"
  # do not create if the file already exists
  if test -f "$csvOutput"; then
    echo "$csvOutput already exists"
  else
    echo "Starting to work with input: $inputname"
    # columns names
    header="configurationID,--boost,--carduse,--decompose,--gluescale,--lkhd,--memlim,--minimize,--prbsimple,--sweepirr,--sweepred,conflicts,cps,reductions"
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