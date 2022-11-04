# Docker container sources 

This directory is a replica of the directory used to build the docker image.

We dropped the inputs videos of the videos directory, but they can be downloaded through the [youtube UGC dataset webpage](https://media.withyoutube.com/).

- original_videos_Animation_480P_Animation_480P-087e
- original_videos_CoverSong_360P_CoverSong_360P-5d20
- original_videos_Gaming_360P_Gaming_360P-56fe
- original_videos_Lecture_360P_Lecture_360P-114f
- original_videos_LiveMusic_360P_LiveMusic_360P-1d94
- original_videos_LyricVideo_360P_LyricVideo_360P-5e87
- original_videos_MusicVideo_360P_MusicVideo_360P-5699
- original_videos_Sports_360P_Sports_360P-4545

We also dropped the log directory, just type 

```mkdir logs```

before actually building the image with 

```sudo docker build .```
