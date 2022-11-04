#!/bin/bash

numb='82'
logfilename="./logs/$numb.log"
inputlocation="$1"
outputlocation="./video$numb.264"

{ time x264  --no-cabac --ref 1 --deblock 1:0:0 --analyse 0x3:0x113 --me hex --subme 4 --no-mixed-ref --merange 16 --trellis 1 --8x8dct --fast-pskip --chroma-qp-offset 0 --bframes 8 --b-pyramid 1 --b-adapt 1 --direct spatial --weightb --weightp 1 --scenecut 40 --rc-lookahead 10 --mbtree --qpmax 69 --aq-mode 1 --output $outputlocation $inputlocation ; } 2> $logfilename
# extract output video size
size=`ls -lrt $outputlocation | awk '{print $5}'`
# analyze log to extract relevant timing information and CPU usage
time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`
# analyze log to extract fps and kbs
persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`
# clean
rm $outputlocation

csvLine="$numb,0,1,1:0:0,0x3:0x113,hex,4,0,16,1,1,1,0,8,1,1,spatial,1,0,1,40,10,1,69,1,"
csvLine="$csvLine$size,$time,$persec"
echo $csvLine