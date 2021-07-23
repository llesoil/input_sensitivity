#!/bin/bash

numb="0"
logfilename="./logs/$numb.log"
inputconf="$1"

./node-15.14.0/out/Release/node --v8-pool-size=0 --preserve-symlinks-main --no-warnings --node-memory-debug  ./node-15.14>
ops=`grep ":*" $logfilename | cut -f2 -d":"`

csvLine="$numb,0,0,0,1,1,1,1,"
csvLine="$csvLine$ops"
echo $csvLine
