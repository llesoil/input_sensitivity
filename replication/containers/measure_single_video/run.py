import json
import os
import subprocess
import sys

video_file = sys.argv[1]
input_cfg = sys.argv[2]

video_name = os.path.splitext(os.path.basename(video_file))[0]
video_results = os.path.join(os.path.dirname(video_file), "logs", video_name + ".csv")
result_header = "configurationID,cabac,ref,deblock,analyse,me,subme,mixed_ref,me_range,trellis,8x8dct,fast_pskip,chroma_qp_offset,bframes,b_pyramid,b_adapt,direct,weightb,open_gop,weightp,scenecut,rc_lookahead,mbtree,qpmax,aq-mode,size,usertime,systemtime,elapsedtime,cpu,frames,fps,kbs"

if not os.path.isfile(video_results):
    open(video_results, "w").write(result_header + "\n")

result_line = subprocess.check_output(
    ["/bin/bash", "./scripts/"+str(input_cfg)+".sh", video_file], universal_newlines=True
)
open(video_results, "a").write(result_line)

result_dict = {
    k.strip(): v.strip()
    for k, v in zip(result_header.split(","), result_line.split(","))
}
print(json.dumps(result_dict))
