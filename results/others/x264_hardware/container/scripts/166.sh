#!/bin/bash

numb='167'
logfilename="./logs/$numb.log"
inputlocation="$1"
outputlocation="./video$numb.264"

{ time x264  --cabac --ref 8 --deblock 1:0:0 --analyse 0x113:0x113 --me dia --subme 7 --mixed-ref --merange 24 --trellis 1 --8x8dct --fast-pskip --chroma-qp-offset 0 --bframes 0 --weightp 0 --scenecut 0 --rc-lookahead 50 --no-mbtree --qpmax 69 --aq-mode 0 --output $outputlocation $inputlocation ; } 2> $logfilename
# extract output video size
size=`ls -lrt $outputlocation | awk '{print $5}'`
# analyze log to extract relevant timing information and CPU usage
time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`
# analyze log to extract fps and kbs
persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`
# clean
rm $outputlocation

csvLine="$numb,1,8,1:0:0,0x113:0x113,dia,7,1,24,1,1,1,0,0,None,None,None,None,None,0,0,50,0,69,0,"
csvLine="$csvLine$size,$time,$persec"
echo $csvLine