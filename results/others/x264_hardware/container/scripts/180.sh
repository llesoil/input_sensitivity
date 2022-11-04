#!/bin/bash

numb='181'
logfilename="./logs/$numb.log"
inputlocation="$1"
outputlocation="./video$numb.264"

{ time x264  --cabac --ref 8 --deblock 0:0:0 --analyse 0x3:0x113 --me umh --subme 9 --mixed-ref --merange 16 --trellis 2 --8x8dct --fast-pskip --chroma-qp-offset -2 --bframes 3 --b-pyramid 2 --b-adapt 2 --direct spatial --weightb --weightp 2 --scenecut 40 --rc-lookahead 10 --mbtree --qpmax 69 --aq-mode 1 --output $outputlocation $inputlocation ; } 2> $logfilename
# extract output video size
size=`ls -lrt $outputlocation | awk '{print $5}'`
# analyze log to extract relevant timing information and CPU usage
time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`
# analyze log to extract fps and kbs
persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`
# clean
rm $outputlocation

csvLine="$numb,1,8,0:0:0,0x3:0x113,umh,9,1,16,2,1,1,-2,3,2,2,spatial,1,0,2,40,10,1,69,1,"
csvLine="$csvLine$size,$time,$persec"
echo $csvLine