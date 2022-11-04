#!/bin/bash

numb='163'
logfilename="./logs/$numb.log"
inputlocation="$1"
outputlocation="./video$numb.264"

{ time x264  --cabac --ref 16 --deblock 1:0:0 --analyse 0x113:0x113 --me umh --subme 6 --mixed-ref --merange 16 --trellis 2 --8x8dct --fast-pskip --chroma-qp-offset -2 --bframes 0 --weightp 0 --scenecut 0 --rc-lookahead 60 --no-mbtree --qpmax 69 --aq-mode 0 --output $outputlocation $inputlocation ; } 2> $logfilename
# extract output video size
size=`ls -lrt $outputlocation | awk '{print $5}'`
# analyze log to extract relevant timing information and CPU usage
time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`
# analyze log to extract fps and kbs
persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`
# clean
rm $outputlocation

csvLine="$numb,1,16,1:0:0,0x113:0x113,umh,6,1,16,2,1,1,-2,0,None,None,None,None,None,0,0,60,0,69,0,"
csvLine="$csvLine$size,$time,$persec"
echo $csvLine