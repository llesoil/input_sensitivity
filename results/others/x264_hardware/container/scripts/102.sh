#!/bin/bash

numb='103'
logfilename="./logs/$numb.log"
inputlocation="$1"
outputlocation="./video$numb.264"

{ time x264  --cabac --ref 2 --deblock 0:0:0 --analyse 0x3:0x3 --me umh --subme 6 --mixed-ref --merange 16 --trellis 1 --8x8dct --fast-pskip --chroma-qp-offset 0 --bframes 8 --b-pyramid 2 --b-adapt 1 --direct spatial --weightb --weightp 1 --scenecut 0 --rc-lookahead 10 --mbtree --qpmax 69 --aq-mode 0 --output $outputlocation $inputlocation ; } 2> $logfilename
# extract output video size
size=`ls -lrt $outputlocation | awk '{print $5}'`
# analyze log to extract relevant timing information and CPU usage
time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`
# analyze log to extract fps and kbs
persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`
# clean
rm $outputlocation

csvLine="$numb,1,2,0:0:0,0x3:0x3,umh,6,1,16,1,1,1,0,8,2,1,spatial,1,0,1,0,10,1,69,0,"
csvLine="$csvLine$size,$time,$persec"
echo $csvLine