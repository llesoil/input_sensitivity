#!/bin/bash

numb='153'
logfilename="./logs/$numb.log"
inputlocation="$1"
outputlocation="./video$numb.264"

{ time x264  --cabac --ref 3 --deblock 1:0:0 --analyse 0x3:0x113 --me hex --subme 8 --mixed-ref --merange 16 --trellis 2 --8x8dct --no-fast-pskip --chroma-qp-offset 0 --bframes 3 --b-pyramid 2 --b-adapt 1 --direct auto --weightb --weightp 2 --scenecut 40 --rc-lookahead 60 --no-mbtree --qpmax 69 --aq-mode 1 --output $outputlocation $inputlocation ; } 2> $logfilename
# extract output video size
size=`ls -lrt $outputlocation | awk '{print $5}'`
# analyze log to extract relevant timing information and CPU usage
time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`
# analyze log to extract fps and kbs
persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`
# clean
rm $outputlocation

csvLine="$numb,1,3,1:0:0,0x3:0x113,hex,8,1,16,2,1,0,0,3,2,1,auto,1,0,2,40,60,0,69,1,"
csvLine="$csvLine$size,$time,$persec"
echo $csvLine