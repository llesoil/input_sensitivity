#!/bin/bash

numb='87'
logfilename="./logs/$numb.log"
inputlocation="$1"
outputlocation="./video$numb.264"

{ time x264  --no-cabac --ref 2 --deblock 1:0:0 --analyse 0:0 --me hex --subme 4 --mixed-ref --merange 24 --trellis 0 --8x8dct --fast-pskip --chroma-qp-offset -2 --bframes 8 --b-pyramid 1 --b-adapt 1 --direct spatial --weightp 1 --scenecut 40 --rc-lookahead 10 --mbtree --qpmax 69 --aq-mode 1 --output $outputlocation $inputlocation ; } 2> $logfilename
# extract output video size
size=`ls -lrt $outputlocation | awk '{print $5}'`
# analyze log to extract relevant timing information and CPU usage
time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`
# analyze log to extract fps and kbs
persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`
# clean
rm $outputlocation

csvLine="$numb,0,2,1:0:0,0:0,hex,4,1,24,0,1,1,-2,8,1,1,spatial,None,0,1,40,10,1,69,1,"
csvLine="$csvLine$size,$time,$persec"
echo $csvLine