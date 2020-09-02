We provide you some instructions, to reproduce the measurements in the same environment :


## 1. Install Docker

If docker is not installed on your laptop, just follow this link; https://docs.docker.com/get-docker/

When it's done, you should be able to run the following command : 

```docker --version```

    >```Docker version 19.03.12, build 48a66213fe```


Useful links if you struggle with docker installation :

- https://www.ibm.com/cloud/blog/installing-docker-windows-fixes-common-problems

- https://medium.com/@randima.somathilaka/docker-engine-install-fails-in-ubuntu-3e70762c2187

- https://runnable.com/docker/install-docker-on-macos


## 2. Pull the image of ubuntu 18.04, launch it

Run the following command;

```sudo docker pull ubuntu:18.04```

Once downloaded, you can launch the image by typing the following command line

```sudo docker run -it ubuntu:18.04 /bin/bash```

Note that this is an ubuntu 18.04.1 version.


### 3. 



First of all, you need to install x264, either by compiling the source code in https://www.videolan.org/developers/x264.html or with the following command line:

<code>sudo apt-get install x264</code>

If you don't have python on your laptop, just run :

<code>sudo apt-get install python-pip</code>

You'll need the pandas library:

<code>pip install pandas</code>

And finally, generate the scripts by typing:

<code>python generate_scripts.py</code>

## LAUNCH:

We let you a test video in the ./data/video folder (called sintel.y4m)

To launch 5 experiments (5*1152 sets of configurations) on it, just copy-paste the following command line in a terminal:

<code>bash launchVideo.sh ./data/video/sintel4.y4m 5 myvideo</code>

If it works, you will see an archive called 'myvideo.tar.gz' in this folder, containing 5 datasets of 1152 lines, each corresponding to a configuration

If you have multiple videos, you can work directly in the output directory (each video has its own folder, the third argument of the command line).

If you want to replicate our project, just get the videos that are present in the video_list.csv file, put them on the ./data/video/ folder and launch:
 
<code>bash multipleVideos.sh</code>


## What is the purpose of each document in this folder?

data -> video -> myvideo.y4m, see https://media.xiph.org/video/derf/ for examples

logs -> x264 logs, log i relative to configuration i

output -> results (.csv)

scripts -> bash scripts, called by launchVideo.sh

x264 -> compiled x264 folder (native x264 works too, but it's an older version) needed nasm version > 2.13

For some videos, the older version can sometimes outperform the new one, it's weird...

generate_scripts.py -> to automate the generation of the scripts in the scripts folder

launchVideo.sh -> to launch experiments, see the launch part

x264_config.csv -> all the videos we tested for this paper

x264_config.csv -> all the configurations we tested for this paper


## Additional advices:

- include a initial set of launches of x264 to "warm" your laptop/server
- close other programs to avoid putting a biais in your measurements


## X264 Version :

To compare your version with ours, open a terminal and launch:

<code>x264 --version</code>

You will get something like this:

x264 0.159.2991 1771b55

built on Jan 29 2020, gcc: 7.4.0

x264 configuration: --chroma-format=all

libx264 configuration: --chroma-format=all

x264 license: GPL version 2 or later



## Hardware :

To compare your hard with ours, open a terminal and launch:

<code>lscpu</code>

You will get something like this:

Architecture:        x86_64

CPU op-mode(s):      32-bit, 64-bit

Byte Order:          Little Endian

CPU(s):              88

On-line CPU(s) list: 0-87

Thread(s) per core:  2

Core(s) per socket:  22

Socket(s):           2

NUMA node(s):        2

Vendor ID:           GenuineIntel

CPU family:          6

Model:               85

Model name:          Intel(R) Xeon(R) Gold 6238 CPU @ 2.10GHz

Stepping:            7

CPU MHz:             1000.695

BogoMIPS:            4200.00

Virtualization:      VT-x

L1d cache:           32K

L1i cache:           32K

L2 cache:            1024K

L3 cache:            30976K

NUMA node0 CPU(s):   0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86

NUMA node1 CPU(s):   1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87

Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts pku ospke avx512_vnni md_clear flush_l1d arch_capabilities
