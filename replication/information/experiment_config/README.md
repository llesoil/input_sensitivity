## experiment_config

A list of files mostly describing the input sensitivity problem, or the experimental protocol:

- categories details the different categories of videos (i.e. the different content of videos)

- commits_inputs_sensitivity presents some interesting commits of x264 showing that developpers of x264 are aware of the input sensitivity problem. See also data, commits_x264.

- config_comparison shows the diffence between our congfigurations and the configurations chosen by Pereira et al. Our configurations are more representative of a real usage of x264, with a larger range of times and sizes.

- documentation and documentation_options detail different characteristics of the chosen p=24 configuration options selected for this experiment

- input_sensitive_example shows an example of three inputs having different performance distribution for encoded sizes. In a journal, we would have presented that as an introduction example.

- introduction_input_sensitivity explains the problem of input sensivity; this article aims at quantifying the influence of inputs on x264's performances. We claim that input data have an influence on software performances, as well as hardware, operating system and software configuration. Ignoring inputs variability may lead to incomplete or non representative results.

- intro_pb is another view of the problem, and completye input_sensitive example with screenshot of videos.

- properties details the different input properties used in this experiment. Since we aren't expert of the field, we extracted description from the original papers and references them.

- table_default_values presents the different default values of configurations

