#!/bin/bash

numb='126'
logfilename="./logs/$numb.log"
inputlocation="$1"
outputlocation="./video$numb.264"

{ time x264  --cabac --ref 3 --deblock 0:0:0 --analyse 0x3:0x3 --me hex --subme 7 --mixed-ref --merange 16 --trellis 1 --no-8x8dct --no-fast-pskip --chroma-qp-offset -2 --bframes 16 --b-pyramid 2 --b-adapt 1 --direct auto --weightb --weightp 2 --scenecut 0 --rc-lookahead 50 --no-mbtree --qpmax 69 --aq-mode 0 --output $outputlocation $inputlocation ; } 2> $logfilename
# extract output video size
size=`ls -lrt $outputlocation | awk '{print $5}'`
# analyze log to extract relevant timing information and CPU usage
time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`
# analyze log to extract fps and kbs
persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`
# clean
rm $outputlocation

csvLine="$numb,1,3,0:0:0,0x3:0x3,hex,7,1,16,1,0,0,-2,16,2,1,auto,1,0,2,0,50,0,69,0,"
csvLine="$csvLine$size,$time,$persec"
echo $csvLine