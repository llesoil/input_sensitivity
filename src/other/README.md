# Other notebooks

- cpu.ipynb, fps.ipynb, encoded_sizes.ipynb & time.ipynb are related to *other performances*.

- L2S.ipynb and Model shifting.ipynb are *old implementations of transfer learning techniques*, see the paper for references.

- since we are not expert of video compression, and to gather some knowledge about x264, we *analyzed 3000+ commits* of (a mirror) [x264 repository](https://github.com/mirror/x264). We use stemming, lemmatization, and searched for keyword ('size', 'mbtree', ect.). We finished the analysis by reading the filtered commits. It led to the commit_input_sensitivity.png graph in results/others/. See requests_commits.ipynb and analyse_commits.ipynb

- In addition to the Mean Opinion Scores collected by the Youtube UGC team, we tried to *find some easy-to-compute metrics* (e.g. using the (r,g,b) histogram of the frames of the video); see Stability_score_video.ipynb

- Notebooks whose names starting with *old_* are just old versions of the code, do not mind them when searching for code
