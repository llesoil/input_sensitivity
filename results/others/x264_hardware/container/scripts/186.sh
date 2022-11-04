#!/bin/bash

numb='187'
logfilename="./logs/$numb.log"
inputlocation="$1"
outputlocation="./video$numb.264"

{ time x264  --no-cabac --ref 16 --deblock 1:0:0 --analyse 0x3:0x113 --me umh --subme 10 --mixed-ref --merange 24 --trellis 2 --8x8dct --fast-pskip --chroma-qp-offset -2 --bframes 8 --b-pyramid 2 --b-adapt 2 --direct spatial --weightb --weightp 2 --scenecut 40 --rc-lookahead 60 --mbtree --qpmax 69 --aq-mode 1 --output $outputlocation $inputlocation ; } 2> $logfilename
# extract output video size
size=`ls -lrt $outputlocation | awk '{print $5}'`
# analyze log to extract relevant timing information and CPU usage
time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`
# analyze log to extract fps and kbs
persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`
# clean
rm $outputlocation

csvLine="$numb,0,16,1:0:0,0x3:0x113,umh,10,1,24,2,1,1,-2,8,2,2,spatial,1,0,2,40,60,1,69,1,"
csvLine="$csvLine$size,$time,$persec"
echo $csvLine