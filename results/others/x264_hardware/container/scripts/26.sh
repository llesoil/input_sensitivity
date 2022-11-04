#!/bin/bash

numb='27'
logfilename="./logs/$numb.log"
inputlocation="$1"
outputlocation="./video$numb.264"

{ time x264  --no-cabac --ref 1 --deblock 0:0:0 --analyse 0:0 --me dia --subme 11 --no-mixed-ref --merange 16 --trellis 0 --no-8x8dct --fast-pskip --chroma-qp-offset 0 --bframes 16 --b-pyramid 2 --b-adapt 1 --weightb --weightp 0 --scenecut 0 --no-mbtree --qpmax 69 --aq-mode 0 --output $outputlocation $inputlocation ; } 2> $logfilename
# extract output video size
size=`ls -lrt $outputlocation | awk '{print $5}'`
# analyze log to extract relevant timing information and CPU usage
time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`
# analyze log to extract fps and kbs
persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`
# clean
rm $outputlocation

csvLine="$numb,0,1,0:0:0,0:0,dia,11,0,16,0,0,1,0,16,2,1,None,1,None,0,0,None,0,69,0,"
csvLine="$csvLine$size,$time,$persec"
echo $csvLine