
### What's the purpose of the notebook?

In this RQ, we study different indicators per software system and per performance property. 
Specifically, we compute the standard deviation of Spearman correlations (as in RQ1 ), the minimal and maximal effects of the most influential option (as in RQ2), the average relative difference of performance due to inputs (as in RQ3). But it is not enough. Intuitively, one needs a way to assess the level of input sensitivity per system and per performance property. We propose a metric that aggregates both indicators of RQ 1, RQ2 and RQ3. In RQ5, we define the score of
Input Sensitivity to estimate the level of IS of a software system.

### Results 

- Input sensitivity is specific to both a configurable system and a performance property. For instance, the sensitivity of x264 configura tions differs depending on whether bitrate or cpu are considered. The related IS score will vary depending on the performance property considered. 

- For high IS scores, researchers should be very careful when experimenting only one input. It is likely that their conclusions will not hold for multiple inputs.
