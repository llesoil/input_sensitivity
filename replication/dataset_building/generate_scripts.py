import pandas as pd

x264_config = pd.read_csv('./x264-config.csv', delimiter=',')

nb_config = x264_config.shape[0]


#default config of x264
default = ['ultrafast','superfast','veryfast','faster','fast','medium',
           'slow', 'slower','veryslow','placebo']
nbdefault = len(default)

#total number of configurations
nbtot = nb_config+nbdefault

#sh numb definition line
numbLines = []

for i in range(nb_tot):
    numbLines.append("numb='"+str(i+1)+"'")


cmd_x264 = []
csvLines = []

for i in range(nb_config):  
    #x264 launch command line
    x264_line = '{ time x264 '
    if x264_config.iloc[i][1]:
        x264_line+=' --no-8x8dct'
    if x264_config.iloc[i][2]:
        x264_line+=' --no-asm'
    if x264_config.iloc[i][3]:
        x264_line+=' --no-cabac'
    if x264_config.iloc[i][4]:
        x264_line+=' --no-deblock'
    if x264_config.iloc[i][5]:
        x264_line+=' --no-fast-pskip'
    if x264_config.iloc[i][6]:
        x264_line+=' --no-mbtree'
    if x264_config.iloc[i][7]:
        x264_line+=' --no-mixed-refs'
    if x264_config.iloc[i][8]:
        x264_line+=' --no-weightb'
    x264_line+=' --rc-lookahead '+str(x264_config.iloc[i][9])
    x264_line+=' --ref '+str(x264_config.iloc[i][10])
    x264_line+=' --output $outputlocation $inputlocation ; } 2> $logfilename'
    cmd_x264.append(x264_line)
    
    # csvLine
    csvLine = 'csvLine="$numb,'
    for val in x264_config.iloc[i].values[1:]:
        csvLine+=str(val)+','
    csvLine+='"'
    csvLines.append(csvLine)  

for i in range(nbdefault):
    x264_line = '{ time x264 --preset '
    x264_line+=default[i]
    x264_line+=' --output $outputlocation $inputlocation ; } 2> $logfilename'
    cmd_x264.append(x264_line)
    csvLines.append('csvLine="$numb,,,,,,,,,,,"')  

assert(len(numbLines)==len(cmd_x264))
assert(len(numbLines)==len(csvLines))

#print(numbLines[0])
#print(cmd_x264[0])
#print(csvLines[0])

"""
#!/bin/bash

numb='1'
logfilename="./logs/$numb.log"
inputlocation="$1"
outputlocation="./video$numb.264"

{ time ./x264/x264 --output $outputlocation $inputlocation ; } 2> $logfilename
# extract output video size
size=`ls -lrt $outputlocation | awk '{print $5}'`
# analyze log to extract relevant timing information and CPU usage
time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`
# analyze log to extract fps and kbs
persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`
# clean
rm $outputlocation

csvLine="$numb,true,true,false,true,true,true,false,true,true,20,1"
csvLine="$csvLine,$size,$time,$persec"
echo $csvLine
"""

for i in range(nb_tot):
    with open('./scripts/'+str(i)+'.sh','w') as f:
        f.write('#!/bin/bash\n\n')
        f.write(numbLines[i])
        f.write('\nlogfilename="./logs/$numb.log"\ninputlocation="$1"\noutputlocation="./video$numb.264"\n\n')
        f.write(cmd_x264[i])
        f.write('\n# extract output video size\n')
        f.write("size=`ls -lrt $outputlocation | awk '{print $5}'`\n")
        f.write('# analyze log to extract relevant timing information and CPU usage\n')
        f.write("""time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`""")
        f.write('\n# analyze log to extract fps and kbs\n')
        f.write("""persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`""")
        f.write('\n# clean\nrm $outputlocation\n\n')
        f.write(csvLines[i])
        f.write('\ncsvLine="$csvLine$size,$time,$persec"\necho $csvLine')
