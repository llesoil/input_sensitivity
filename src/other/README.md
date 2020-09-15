# Other notebooks

- bitrate_algorithm_comparison.ipynb and the related bitrate_algorithm_comparison.md details why we chose random forest, instead of a linear regression or a neural network.

- cpu.ipynb, fps.ipynb, encoded_sizes.ipynb & time.ipynb are related to *other performances*.

- L2S.ipynb and Model shifting.ipynb are *implementations of transfer learning techniques*, but we didn't have the time to compare them with inputec for now. Future work! :)  No_transfer.ipynb is a simple learning method (i.e. without transfer).

- since we are not expert of video compression, and to gather some knowledge about x264, we *analyzed 3000+ commits* of (a mirror) [x264 repository](https://github.com/mirror/x264). We use stemming, lemmatization, and searched for keyword ('size', 'mbtree', ect.). We finished the analysis by reading the filtered commits. It led to the commit_input_sensitivity.png graph in results/others/. See requests_commits.ipynb and analyse_commits.ipynb

- In addition to the Mean Opinion Scores collected by the Youtube UGC team, we tried to *find some easy-to-compute metrics* (e.g. using the (r,g,b) histogram of the frames of the video); see Stability_score_video.ipynb
