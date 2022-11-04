#!/bin/bash

numb='90'
logfilename="./logs/$numb.log"
inputlocation="$1"
outputlocation="./video$numb.264"

{ time x264  --no-cabac --ref 2 --deblock 1:0:0 --analyse 0x3:0x113 --me hex --subme 4 --no-mixed-ref --merange 16 --trellis 0 --8x8dct --no-fast-pskip --chroma-qp-offset -2 --bframes 0 --weightp 0 --scenecut 0 --no-mbtree --qpmax 69 --aq-mode 1 --output $outputlocation $inputlocation ; } 2> $logfilename
# extract output video size
size=`ls -lrt $outputlocation | awk '{print $5}'`
# analyze log to extract relevant timing information and CPU usage
time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`
# analyze log to extract fps and kbs
persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`
# clean
rm $outputlocation

csvLine="$numb,0,2,1:0:0,0x3:0x113,hex,4,0,16,0,1,0,-2,0,None,None,None,None,None,0,0,None,0,69,1,"
csvLine="$csvLine$size,$time,$persec"
echo $csvLine