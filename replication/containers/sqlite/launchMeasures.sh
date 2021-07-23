#!/bin/bash

#set -xv

# for each input (here a database file)
cat listInputs.csv | while read line
do
  read -d, inputname < <(echo $line)
  path="./inputs/$inputname.pdf"
  bash init_db.sh $inputname
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
    header="configurationID,-deserialize,-memtrace,-maxsize,-append,-output,time0,time1,time2,time3,time4,time5,time6,time7,time8,time9,time10,time11,time12,time13,time14,time15"
    touch $csvOutput
    #cat /dev/null > $csvOutput
    echo "$header" >> $csvOutput
    # we execute all the runtime configs
    for config in $configs
    do
      echo "Processing: $config"
      csvLine=`bash $config > test.txt`
      # and store the line in the csv
      awk 'END {print}' test.txt >> $csvOutput
    done
fi
done

echo "Done with input $inputname"
