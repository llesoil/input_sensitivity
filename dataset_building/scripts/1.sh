#!/bin/bash

numb='2'
logfilename="./logs/$numb.log"
inputlocation="$1"
outputlocation="./video$numb.264"

{ time ./x264/x264  --no-8x8dct --no-cabac --no-fast-pskip --no-weightb --rc-lookahead 40 --ref 9 --output $outputlocation $inputlocation ; } 2> $logfilename
# extract output video size
size=`ls -lrt $outputlocation | awk '{print $5}'`
# analyze log to extract relevant timing information and CPU usage
time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`
# analyze log to extract fps and kbs
persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`
# clean
rm $outputlocation

csvLine="$numb,True,False,True,False,True,False,False,True,40,9,"
csvLine="$csvLine$size,$time,$persec"
echo $csvLine