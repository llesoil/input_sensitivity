#!/bin/bash

numb='99'
logfilename="./logs/$numb.log"
inputlocation="$1"
outputlocation="./video$numb.264"

{ time x264  --cabac --ref 5 --deblock 1:0:0 --analyse 0x3:0x113 --me hex --subme 6 --mixed-ref --merange 16 --trellis 1 --8x8dct --no-fast-pskip --chroma-qp-offset -2 --bframes 3 --b-adapt 1 --direct spatial --weightb --weightp 1 --scenecut 40 --rc-lookahead 30 --mbtree --qpmax 69 --aq-mode 1 --output $outputlocation $inputlocation ; } 2> $logfilename
# extract output video size
size=`ls -lrt $outputlocation | awk '{print $5}'`
# analyze log to extract relevant timing information and CPU usage
time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`
# analyze log to extract fps and kbs
persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`
# clean
rm $outputlocation

csvLine="$numb,1,5,1:0:0,0x3:0x113,hex,6,1,16,1,1,0,-2,3,None,1,spatial,1,0,1,40,30,1,69,1,"
csvLine="$csvLine$size,$time,$persec"
echo $csvLine