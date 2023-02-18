
The different results are dispatched in the different sub-directories, and detailed in each related README file :
- in the **figures** directory, we put the images that are used directly in the publication, ordered following their apparition in the document. For instance, figure1.png is related to the Figure 1 of the article (the pdf in the root directory). 
- in the **others** directory, we put other small experiments not directly related to the main results, but complementing the original research questions
- in the **RQs** directory, we put the results direclty associated to the different research questions presented as part of the paper
- in the **splconqueror** directory, we put the results of an experiment conducted with SPLConqueror
- in the **systems** directory, we put the results of the research question on a per-system basis, to see the specifics of each software system and to be able to compare systems between each other. All images are directly generated from the RQs notebooks.


|   **Location** | **Related RQ or XP** |  **Content**  |  **Goal**  |  **Key Result**  |
| others/config/sampling | Measurement process |  Statistical tests | Ensure the uniformity of the sampling of chosen configurations | Sampling is good enough |
| others/x264_groups | RQ4 | Clustering algorithms | Highlight groups of input videos sharing similar performance profiles for x264 | Domain profiles match performance profiles |
|  others/x264_hardware | threats to validity | Replication of measurements & xps with different hardware platforms | Ensure the robustness of results | We can expect our results to be valid whatever the hardware platform |
|  others/x264_x265 | threats to validity | Replication of measurements & xps with different software systems of the same domain | Ensure the robustness of results | We can expect our results to be valid whatever the chosen software |
|  RQs/RQ1  |  RQ1  | Spearman Correlation | To what extent are the performance distributions of configurable systems changing with input data? | There exist different performance profiles according to the considered inputs |
|  RQs/RQ2  |  RQ2  | Feature Importance & Regression Coefficient | To what extent the effects of configuration options are consistent with input data? | The influence and effects of software options vary with input data |
|  RQs/RQ3  |  RQ3 | Performance Ratios | How much performance are lost when reusing a configuration across inputs? | We can gain/lose ~25% when blindly reusing a configuration across different inputs |
|  RQs/RQ4  |  RQ4 | Clustering & Interpretability of clusters | What is the benefit of grouping the inputs and how to characterise the different groups? | Based on the input specifications, we can estimate how to put inputs in performance groups, without performance measurements |
|  RQs/RQ5  |  Score of input sensitivity | Compute a simple math formula | How to estimate whether a software system is input-sensitive (or not)? | Based on the previous question, we compute the score and it is relevant wrt other research papers conclusions |
|  RQs/RQ6  |  SLR | Read research papers & answer questions | How do state-of-the-art papers address input sensitivity? | Input sensitivity could be more systematically addressed in research, as proposed in our paper |
|  splconqueror  |  External tool testing | Experiment with other tools | Ensure the robustness of the results | We retrieve our results with SPLConqueror |
