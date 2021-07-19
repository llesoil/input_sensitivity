
# RQ4 - Input sensitivity in research

This part addresses the following question :

## How do state-of-the-art papers address input sensitivity?

To do so, we gather research papers (see the related submission for details).

We read each of them carefully and  answer  four  different  questions:

### Q1  -  Do  the  software systems process input data?

If most of research papers do not study configurable systems processing input data, the impact of input sensitivity in research would be relatively low. The idea of this research question is to estimate which proportion of the performance models could be affected by input sensitivity. 

### Q2 - Are the performance models applied on several inputs?

Then, we check if the research papers include several inputs in the study.  If  not,  it  would  suggest  that  the  performance  model only captures a partial truth, and might not generalize for other inputs fed to the software system. 

### Q3 - Is the problem of input sensitivity mentioned e.g., in threat? 

This question aims to state whether researchers are aware of the input sensitivity threat, and which proportion of the papers mention it as a potential threat to validity. 

### Q4 - Does the paper propose a solution to generalize the performance model across inputs? 

Finally, we check if the paper proposes a solution solving input sensitivity i.e., if they are able to train a performance model that is general and robust enough to predict a near-optimal configuration for any input. 

The results were obtained by one author and validated by two authors of this paper. 

Feel free to contact us if you disagree with the results!

## RESULTS

The rest of this document details the results for each paper and justifies the answers to Q1, Q2, Q3 and Q4. 

We also provide references for each paper (bibTeX), and they are available in the sub-folder "papers" of this directory. For the paper with the id 12, search for the file named "12.pdf".

### Paper 1

Title : 
Data-Efficient Performance Learning for Configurable Systems

Bibtex :
@article{Guo2017,
  doi = {10.1007/s10664-017-9573-6},
  url = {https://doi.org/10.1007/s10664-017-9573-6},
  year = {2017},
  month = nov,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {23},
  number = {3},
  pages = {1826--1867},
  author = {Jianmei Guo and Dingyu Yang and Norbert Siegmund and Sven Apel and Atrisha Sarkar and Pavel Valov and Krzysztof Czarnecki and Andrzej Wasowski and Huiqun Yu},
  title = {Data-efficient performance learning for configurable systems},
  journal = {Empirical Software Engineering}
}

Q1 : 
Yes, 10 configurable systems, including x264, SQLite, llvm, etc.

Q2 : 
"We measured their performance using standard benchmarks from the respective application domain." -> but looking at the material  https://github.com/jmguo/DECART/blob/master/multipleMetrics/results/CART_x264_Metric1and2_Details_crossvalidation_gridsearch.csv
seems to be only one input per system -> No

Q3 : 
"We measured their performance using standard benchmarks from the respective application domain." in threats, but no mention of a potential lack of generalization -> No

Q4 : 
No


### Paper 2

Title : 
Transfer learning for improving model predictions in highly configurable software

Bibtex :
@inproceedings{10.1109/SEAMS.2017.11,
author = {Jamshidi, Pooyan and Velez, Miguel and K\"{a}stner, Christian and Siegmund, Norbert and Kawthekar, Prasad},
title = {Transfer Learning for Improving Model Predictions in Highly Configurable Software},
year = {2017},
isbn = {9781538615508},
publisher = {IEEE Press},
url = {https://doi.org/10.1109/SEAMS.2017.11},
doi = {10.1109/SEAMS.2017.11},
abstract = {Modern software systems are built to be used in dynamic environments using configuration
capabilities to adapt to changes and external uncertainties. In a self-adaptation
context, we are often interested in reasoning about the performance of the systems
under different configurations. Usually, we learn a black-box model based on real
measurements to predict the performance of the system given a specific configuration.
However, as modern systems become more complex, there are many configuration parameters
that may interact and we end up learning an exponentially large configuration space.
Naturally, this does not scale when relying on real measurements in the actual changing
environment. We propose a different solution: Instead of taking the measurements from
the real system, we learn the model using samples from other sources, such as simulators
that approximate performance of the real system at low cost. We define a cost model
that transform the traditional view of model learning into a multi-objective problem
that not only takes into account model accuracy but also measurements effort as well.
We evaluate our cost-aware transfer learning solution using real-world configurable
software including (i) a robotic system, (ii) 3 different stream processing applications,
and (iii) a NoSQL database system. The experimental results demonstrate that our approach
can achieve (a) a high prediction accuracy, as well as (b) a high model reliability.},
booktitle = {Proceedings of the 12th International Symposium on Software Engineering for Adaptive and Self-Managing Systems},
pages = {31–41},
numpages = {11},
keywords = {model prediction, model learning, machine learning, transfer learning, highly configurable software},
location = {Buenos Aires, Argentina},
series = {SEAMS '17}
}

Q1 : 
Yes, Cassandra, nosql database

Q2 : 
Yes, see Table 1 in the supplementary material here https://github.com/pooyanjamshidi/transferlearning/blob/master/online-appendix.pdf

Q3 : 
In the article, section Threats to validity "Moreover,
we used standard benchmarks so that we are confident in that
we have measured a realistic scenario." -> Yes

Q4 : 
While the proposed method is promising to apply on input sensitivity pb, it is not directly applied to inputs -> No


### Paper 3

Title : 
Transfer learning for performance modeling of configurable systems: an exploratory analysis

Bibtex :
@inproceedings{10.5555/3155562.3155625,
author = {Jamshidi, Pooyan and Siegmund, Norbert and Velez, Miguel and K\"{a}stner, Christian and Patel, Akshay and Agarwal, Yuvraj},
title = {Transfer Learning for Performance Modeling of Configurable Systems: An Exploratory Analysis},
year = {2017},
isbn = {9781538626849},
publisher = {IEEE Press},
abstract = { Modern software systems provide many configuration options which significantly influence
their non-functional properties. To understand and predict the effect of configuration
options, several sampling and learning strategies have been proposed, albeit often
with significant cost to cover the highly dimensional configuration space. Recently,
transfer learning has been applied to reduce the effort of constructing performance
models by transferring knowledge about performance behavior across environments. While
this line of research is promising to learn more accurate models at a lower cost,
it is unclear why and when transfer learning works for performance modeling. To shed
light on when it is beneficial to apply transfer learning, we conducted an empirical
study on four popular software systems, varying software configurations and environmental
conditions, such as hardware, workload, and software versions, to identify the key
knowledge pieces that can be exploited for transfer learning. Our results show that
in small environmental changes (e.g., homogeneous workload change), by applying a
linear transformation to the performance model, we can understand the performance
behavior of the target environment, while for severe environmental changes (e.g.,
drastic workload change) we can transfer only knowledge that makes sampling more efficient,
e.g., by reducing the dimensionality of the configuration space. },
booktitle = {Proceedings of the 32nd IEEE/ACM International Conference on Automated Software Engineering},
pages = {497–508},
numpages = {12},
keywords = {Performance analysis, transfer learning},
location = {Urbana-Champaign, IL, USA},
series = {ASE 2017}
}

Q1 : 
Yes, 4 configurable systems, including x264 and SQLite

Q2 : 
Yes, different workloads, see |W| in Table 1

Q3 : 
Yes, "We selected a diverse set of subject
systems and a large number of purposefully selected environ-
ment changes, but, as usual, one has to be careful when gen-
eralizing to other subject systems and environment changes." in threats section 

Q4 : 
Yes, the paper uses transfer learning to transfer performances from one environment to the others, which includes inputs as well.
Indeed a promising technique to test on more inputs.

(AM note: "unfortunately", they sometimes mix inputs' changes with versions' changes and hardwares' changes)

### Paper 4

Title : 
Finding near-optimal configurations in product lines by random sampling

Bibtex :
@inproceedings{10.1145/3106237.3106273,
author = {Oh, Jeho and Batory, Don and Myers, Margaret and Siegmund, Norbert},
title = {Finding Near-Optimal Configurations in Product Lines by Random Sampling},
year = {2017},
isbn = {9781450351058},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3106237.3106273},
doi = {10.1145/3106237.3106273},
abstract = { Software Product Lines (SPLs) are highly configurable systems. This raises the challenge
to find optimal performing configurations for an anticipated workload. As SPL configuration
spaces are huge, it is infeasible to benchmark all configurations to find an optimal
one. Prior work focused on building performance models to predict and optimize SPL
configurations. Instead, we randomly sample and recursively search a configuration
space directly to find near-optimal configurations without constructing a prediction
model. Our algorithms are simpler and have higher accuracy and efficiency. },
booktitle = {Proceedings of the 2017 11th Joint Meeting on Foundations of Software Engineering},
pages = {61–71},
numpages = {11},
keywords = {searching configuration spaces, finding optimal configurations, software product lines},
location = {Paderborn, Germany},
series = {ESEC/FSE 2017}
}

Q1 : 
Yes, x264 and llvm at least

Q2 : 
In threats "We used ground-truth data of [30] which are
measurements of real systems."
Following the link in ref 30, http://fosd.de/SPLConqueror
it is yet unclear which data are used for this paper.
Assuming the data are those from the ICSE 2012 paper, there is only one input per system when downloading the supplementary material -> No

Q3 : 
No mention of input sensitivity, No

Q4 : 
No

### Paper 5

Title : 
Tradeoffs in modeling performance of highly configurable software systems

Bibtex :
@article{Kolesnikov2018,
  doi = {10.1007/s10270-018-0662-9},
  url = {https://doi.org/10.1007/s10270-018-0662-9},
  year = {2018},
  month = feb,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {18},
  number = {3},
  pages = {2265--2283},
  author = {Sergiy Kolesnikov and Norbert Siegmund and Christian K\"{a}stner and Alexander Grebhahn and Sven Apel},
  title = {Tradeoffs in modeling performance of highly configurable software systems},
  journal = {Software {\&} Systems Modeling}
}

Q1 : 
Yes, 10 real-world highly-configurable systems, including x264

Q2 : 
According to the paper section 3.3 "using standard benchmarks for the respective domain", but following https://www.se.cs.uni-saarland.de/projects/tradeoffs/
only one input per system ? -> No

Q3 : 
No mention of input sensitivity, No

Q4 : 
No

### Paper 6

Title : 
Using bad learners to find good configurations

Bibtex :
@inproceedings{10.1145/3106237.3106238,
author = {Nair, Vivek and Menzies, Tim and Siegmund, Norbert and Apel, Sven},
title = {Using Bad Learners to Find Good Configurations},
year = {2017},
isbn = {9781450351058},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3106237.3106238},
doi = {10.1145/3106237.3106238},
abstract = { Finding the optimally performing configuration of a software system for a given setting
is often challenging. Recent approaches address this challenge by learning performance
models based on a sample set of configurations. However, building an accurate performance
model can be very expensive (and is often infeasible in practice). The central insight
of this paper is that exact performance values (e.g., the response time of a software
system) are not required to rank configurations and to identify the optimal one. As
shown by our experiments, performance models that are cheap to learn but inaccurate
(with respect to the difference between actual and predicted performance) can still
be used rank configurations and hence find the optimal configuration. This novel rank-based
approach allows us to significantly reduce the cost (in terms of number of measurements
of sample configuration) as well as the time required to build performance models.
We evaluate our approach with 21 scenarios based on 9 software systems and demonstrate
that our approach is beneficial in 16 scenarios; for the remaining 5 scenarios, an
accurate model can be built by using very few samples anyway, without the need for
a rank-based approach. },
booktitle = {Proceedings of the 2017 11th Joint Meeting on Foundations of Software Engineering},
pages = {257–267},
numpages = {11},
keywords = {Rank-based method, Performance Prediction, Sampling, SBSE},
location = {Paderborn, Germany},
series = {ESEC/FSE 2017}
}

Q1 : 
Yes, 9 software systems including berkeley lrzip, etc.

Q2 : 

Depending on the software, yes or no, e.g.

No for SQLite
"For SQLite, we cannot measure all possible configurations inreasonable time. Hence, we sampled only 100 configurations tocompare prediction and actual values. We are aware that this evalu-ation leaves room for outliers and that measurement bias can causefalse interpretations [11]. Since we limit our attention to predictingperformance for a given workload, we did not vary benchmarks."

Yes for Apache Storm, it considers several inputs
"The experiment considers three benchmarks namely:
WordCount (wc) counts the number of occurences of thewords in a text file.
RollingSort (rs) implements a common pattern in real-timeanalysis that performs rolling counts of messages.
SOL (sol) is a network intensive topology, where the mes-sage is routed through an inter-worker network"

We choose to answer "Yes" for Q2.

Q3 : 
No mention of input sensitivity, No

Q4 : 
No

### Paper 7

Title : 
Finding Faster Configurations using FLASH

Bibtex :
@ARTICLE{8469102,
author={Nair, Vivek and Yu, Zhe and Menzies, Tim and Siegmund, Norbert and Apel, Sven},
journal={IEEE Transactions on Software Engineering}, 
title={Finding Faster Configurations Using <sc>FLASH</sc>}, 
year={2020},
volume={46},
number={7},
pages={794-811},
doi={10.1109/TSE.2018.2870895}
}

Q1 : 
Yes, including x264, llvm, etc.

Q2 : 
Word Count and Rolling Sort executed with Stream Processing Systems
Not many inputs, but we can answer Yes

Q3 : 
"Hence,  there  is  noinherent  mechanism  in  FLASH which  would  adapt  itself based  on  the  change  in  workload.  This  non-stationary nature  of  the  problem  is  a  significant  assumption  and currently  not  addressed  in  this  paper."
-> Yes

Q4 : 
Same citation as Q3, the paper is honest on the limitations of the approach.
-> No

### Paper 8

Title : 
Measuring Energy Consumption for Web Service Product Configuration

Bibtex :
@inproceedings{10.1145/2684200.2684314,
author = {Murwantara, I Made and Bordbar, Behzad and Minku, Leandro L.},
title = {Measuring Energy Consumption for Web Service Product Configuration},
year = {2014},
isbn = {9781450330015},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/2684200.2684314},
doi = {10.1145/2684200.2684314},
abstract = {Because of the economies of scale that Cloud provides, there is great interest in
hosting web services on the Cloud. Web services are created from components such as
Database Management Systems and HTTP servers. There is a wide variety of components
that can be used to configure a web service. The choice of components influences the
performance and energy consumption. Most current research in the web service technologies
focuses on system performance, and only small number of researchers give attention
to energy consumption. In this paper, we propose a method to select the web service
configurations which reduce energy consumption. Our method has capabilities to manage
feature configuration and predict energy consumption of web service systems. To validate,
we developed a technique to measure energy consumption of several web service configurations
running in a Virtualized environment. Our approach allows Cloud companies to provide
choices of web service technology that consumes less energy.},
booktitle = {Proceedings of the 16th International Conference on Information Integration and Web-Based Applications & Services},
pages = {224–228},
numpages = {5},
keywords = {Machine Learning, Energy Aware, Web System, Software Product Line},
location = {Hanoi, Viet Nam},
series = {iiWAS '14}
}

Q1 :
Yes, Apache, Nginx, and combination of tools that takes an input (web content)

Q2 : 
Yes, e.g.
"Further, we use Machine Learning technique to pre-dict individual component and configuration energy usagewith varied workload"


Q3 : 
Yes, they even measure different performance values for different values, which is highly valuable in terms of input sensitivity.
"This gives the insight that, depending on the workload, different configurations are unlikely to lead to significant impact on CPU power consumption."
6.1.2 Experimental Results is really an interesting case that carry the same message as our paper.

Q4 : 
Yes, they highlight "levels" of workloads, and based on that, they can predict an accurate energy consumption associated to the workload.


### Paper 9

Title : 
Using machine learning to infer constraints for product lines

Bibtex :
@inproceedings{Temple2016,
  doi = {10.1145/2934466.2934472},
  url = {https://doi.org/10.1145/2934466.2934472},
  year = {2016},
  month = sep,
  publisher = {{ACM}},
  author = {Paul Temple and Jos{\'{e}} A. Galindo and Mathieu Acher and Jean-Marc J{\'{e}}z{\'{e}}quel},
  title = {Using machine learning to infer constraints for product lines},
  booktitle = {Proceedings of the 20th International Systems and Software Product Line Conference}
}

Q1 : 
No, the video generator do not consider any input data

Q2 : 
Q1 = No

Q3 : 
Q1 = No

Q4 : 
Q1 = No

### Paper 10

Title : 
Learning Contextual-Variability Models

Bibtex :
@ARTICLE{8106868,
author={Temple, Paul and Acher, Mathieu and Jézéquel, Jean-Marc and Barais, Olivier},
journal={IEEE Software}, 
title={Learning Contextual-Variability Models}, 
year={2017},
volume={34},
number={6},
pages={64-70},
doi={10.1109/MS.2017.4121211}
} 

Q1 : 
Yes, 10 systems, including x264, SQLite

Q2 : 
One input per system, see https://github.com/learningconstraints/ICSE-17/tree/master/datasets
following the link of http://learningconstraints.github.io/
-> No

Q3 : 
"A first difficulty is related to the development of procedures (oracles) for measuring software configurations in different contexts. It may be difficult to find the right data or to create the realistic contextual conditions"
-> Yes

Q4 : 
No


### Paper 11

Title : 
Transferring Performance Prediction Models Across Different Hardware Platforms

Bibtex :
@inproceedings{10.1145/3030207.3030216,
author = {Valov, Pavel and Petkovich, Jean-Christophe and Guo, Jianmei and Fischmeister, Sebastian and Czarnecki, Krzysztof},
title = {Transferring Performance Prediction Models Across Different Hardware Platforms},
year = {2017},
isbn = {9781450344043},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3030207.3030216},
doi = {10.1145/3030207.3030216},
abstract = {Many software systems provide configuration options relevant to users, which are often
called features. Features influence functional properties of software systems as well
as non-functional ones, such as performance and memory consumption. Researchers have
successfully demonstrated the correlation between feature selection and performance.
However, the generality of these performance models across different hardware platforms
has not yet been evaluated.We propose a technique for enhancing generality of performance
models across different hardware environments using linear transformation. Empirical
studies on three real-world software systems show that our approach is computationally
efficient and can achieve high accuracy (less than 10% mean relative error) when predicting
system performance across 23 different hardware platforms. Moreover, we investigate
why the approach works by comparing performance distributions of systems and structure
of performance models across different platforms.},
booktitle = {Proceedings of the 8th ACM/SPEC on International Conference on Performance Engineering},
pages = {39–50},
numpages = {12},
keywords = {performance modelling, model transfer, regression trees, linear transformation},
location = {L'Aquila, Italy},
series = {ICPE '17}
}

Q1 : 
Yes, including x264, xz and sqlite

Q2 : 
No, they just make the hardware vary.

Q3 : 
In future work, 
"We  suspect  that  variations in system workload might influence transferability of system’sperformance prediction model by distorting system’s performance distribution across different hardware environments."
-> Yes + plus an interesting idea to combine hardware and input data

Q4 : 
No


### Paper 12

Title : 
Optimal Reconfiguration of Dynamic Software Product Lines Based on Performance-Influence Models

Bibtex :
@inproceedings{10.1145/3233027.3233030,
author = {Weckesser, Markus and Kluge, Roland and Pfannem\"{u}ller, Martin and Matth\'{e}, Michael and Sch\"{u}rr, Andy and Becker, Christian},
title = {Optimal Reconfiguration of Dynamic Software Product Lines Based on Performance-Influence Models},
year = {2018},
isbn = {9781450364645},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3233027.3233030},
doi = {10.1145/3233027.3233030},
abstract = {Today's adaptive software systems (i) are often highly configurable product lines,
exhibiting hundreds of potentially conflicting configuration options; (ii) are context
dependent, forcing the system to reconfigure to ever-changing contextual situations
at runtime; (iii) need to fulfill context-dependent performance goals by optimizing
measurable nonfunctional properties. Usually, a large number of consistent configurations
exists for a given context, and each consistent configuration may perform differently
with regard to the current context and performance goal(s). Therefore, it is crucial
to consider nonfunctional properties for identifying an appropriate configuration.
Existing black-box approaches for estimating the performance of configurations provide
no means for determining context-sensitive reconfiguration decisions at runtime that
are both consistent and optimal, and hardly allow for combining multiple context-dependent
quality goals. In this paper, we propose a comprehensive approach based on Dynamic
Software Product Lines (DSPL) for obtaining consistent and optimal reconfiguration
decisions. We use training data obtained from simulations to learn performance-influence
models. A novel integrated runtime representation captures both consistency properties
and the learned performance-influence models. Our solution provides the flexibility
to define multiple context-dependent performance goals. We have implemented our approach
as a standalone component. Based on an Internet-of-Things case study using adaptive
wireless sensor networks, we evaluate our approach with regard to effectiveness, efficiency,
and applicability.},
booktitle = {Proceedings of the 22nd International Systems and Software Product Line Conference - Volume 1},
pages = {98–109},
numpages = {12},
keywords = {dynamic software product lines, performance-influence models, machine learning},
location = {Gothenburg, Sweden},
series = {SPLC '18}
}

Q1 : 
No, the simulator do not consider any input data

Q2 : 
Q1 = No

Q3 : 
Q1 = No

Q4 : 
Q1 = No

### Paper 13

Title : 
VaryLATEX: Learning Paper Variants That Meet Constraints

Bibtex :
@inproceedings{10.1145/3168365.3168372,
author = {Acher, Mathieu and Temple, Paul and J\'{e}z\'{e}quel, Jean-Marc and Galindo, Jos\'{e} A. and Martinez, Jabier and Ziadi, Tewfik},
title = {VaryLATEX: Learning Paper Variants That Meet Constraints},
year = {2018},
isbn = {9781450353984},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3168365.3168372},
doi = {10.1145/3168365.3168372},
abstract = {How to submit a research paper, a technical report, a grant proposal, or a curriculum
vitae that respect imposed constraints such as formatting instructions and page limits?
It is a challenging task, especially when coping with time pressure. In this work,
we present VaryLATEX, a solution based on variability, constraint programming, and
machine learning techniques for documents written in LATEX to meet constraints and
deliver on time. Users simply have to annotate LATEX source files with variability
information, e.g., (de)activating portions of text, tuning figures' sizes, or tweaking
line spacing. Then, a fully automated procedure learns constraints among Boolean and
numerical values for avoiding non-acceptable paper variants, and finally, users can
further configure their papers (e.g., aesthetic considerations) or pick a (random)
paper variant that meets constraints, e.g., page limits. We describe our implementation
and report the results of two experiences with VaryLATEX.},
booktitle = {Proceedings of the 12th International Workshop on Variability Modelling of Software-Intensive Systems},
pages = {83–88},
numpages = {6},
keywords = {variability modelling, machine learning, constraint programming, generators, technical writing, LATEX},
location = {Madrid, Spain},
series = {VAMOS 2018}
}

Q1 : 
Yes, the software process different scripts of latex code

Q2 : 
Yes, different pdf codes are fed to the system

Q3 : 
No ?

Q4 : 
Yes, different pdf codes are fed to the system

### Paper 15

Title : 
Cost-Efficient Sampling for Performance Predictionof Configurable Systems

Bibtex :
@inproceedings{10.1109/ASE.2015.45,
author = {Sarkar, Atri and Guo, Jianmei and Siegmund, Norbert and Apel, Sven and Czarnecki, Krzysztof},
title = {Cost-Efficient Sampling for Performance Prediction of Configurable Systems},
year = {2015},
isbn = {9781509000241},
publisher = {IEEE Press},
url = {https://doi.org/10.1109/ASE.2015.45},
doi = {10.1109/ASE.2015.45},
abstract = {A key challenge of the development and maintenance of configurable systems is to predict
the performance of individual system variants based on the features selected. It is
usually infeasible to measure the performance of all possible variants, due to feature
combinatorics. Previous approaches predict performance based on small samples of measured
variants, but it is still open how to dynamically determine an ideal sample that balances
prediction accuracy and measurement effort. In this paper, we adapt two widely-used
sampling strategies for performance prediction to the domain of configurable systems
and evaluate them in terms of sampling cost, which considers prediction accuracy and
measurement effort simultaneously. To generate an initial sample, we introduce a new
heuristic based on feature frequencies and compare it to a traditional method based
on t-way feature coverage. We conduct experiments on six real-world systems and provide
guidelines for stakeholders to predict performance by sampling.},
booktitle = {Proceedings of the 30th IEEE/ACM International Conference on Automated Software Engineering},
pages = {342–352},
numpages = {11},
location = {Lincoln, Nebraska},
series = {ASE '15}
}

Q1 : 
Yes, 6 systems, including x264, llvm, SQLite, etc.

Q2 : 
No, they focus on configuration options (they are mostly interested in varying the sample size) so it explains why

Q3 : 
No

Q4 : 
No

### Paper 15

Title : 
Towards Adversarial Configurations for Software Product Lines

Bibtex :
@misc{temple2018adversarial,
      title={Towards Adversarial Configurations for Software Product Lines}, 
      author={Paul Temple and Mathieu Acher and Battista Biggio and Jean-Marc Jézéquel and Fabio Roli},
      year={2018},
      eprint={1805.12021},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

Q1 : 
No, the video generator do not consider any input data

Q2 : 
Q1 = No

Q3 : 
Q1 = No

Q4 : 
Q1 = No

### Paper 16

Title :
Faster Discovery of Faster System Configurations with Spectral Learning

Bibtex :
@article{10.1007/s10515-017-0225-2,
author = {Nair, Vivek and Menzies, Tim and Siegmund, Norbert and Apel, Sven},
title = {Faster Discovery of Faster System Configurations with Spectral Learning},
year = {2018},
issue_date = {June      2018},
publisher = {Kluwer Academic Publishers},
address = {USA},
volume = {25},
number = {2},
issn = {0928-8910},
url = {https://doi.org/10.1007/s10515-017-0225-2},
doi = {10.1007/s10515-017-0225-2},
abstract = {Despite the huge spread and economical importance of configurable software systems,
there is unsatisfactory support in utilizing the full potential of these systems with
respect to finding performance-optimal configurations. Prior work on predicting the
performance of software configurations suffered from either (a) requiring far too
many sample configurations or (b) large variances in their predictions. Both these
problems can be avoided using the WHAT spectral learner. WHAT's innovation is the
use of the spectrum (eigenvalues) of the distance matrix between the configurations
of a configurable software system, to perform dimensionality reduction. Within that
reduced configuration space, many closely associated configurations can be studied
by executing only a few sample configurations. For the subject systems studied here,
a few dozen samples yield accurate and stable predictors--less than 10% prediction
error, with a standard deviation of less than 2%. When compared to the state of the
art, WHAT (a) requires 2---10 times fewer samples to achieve similar prediction accuracies,
and (b) its predictions are more stable (i.e., have lower standard deviation). Furthermore,
we demonstrate that predictive models generated by WHAT can be used by optimizers
to discover system configurations that closely approach the optimal performance.},
journal = {Automated Software Engg.},
month = jun,
pages = {247–277},
numpages = {31},
keywords = {Sampling, Decision trees, Search-based software engineering, Performance prediction, Spectral learning}
}

Q1 : 
Yes, including x264, SQLite, Apache

Q2 : 
 -> SQLite
Only one input per system e.g. Sintel for x264, the test suite (i.e. without detailing each test) for llvm, etc.
-> No

Q3 : 
No

Q4 : 
"Since we aim at predicting performance for a special workload, we do not have to vary benchmarks"
-> No


### Paper 17

Title : 
Performance-Influence Models for Highly Configurable Systems

Bibtex :
@inproceedings{10.1145/2786805.2786845,
author = {Siegmund, Norbert and Grebhahn, Alexander and Apel, Sven and K\"{a}stner, Christian},
title = {Performance-Influence Models for Highly Configurable Systems},
year = {2015},
isbn = {9781450336758},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/2786805.2786845},
doi = {10.1145/2786805.2786845},
abstract = { Almost every complex software system today is configurable. While configurability
has many benefits, it challenges performance prediction, optimization, and debugging.
Often, the influences of individual configuration options on performance are unknown.
Worse, configuration options may interact, giving rise to a configuration space of
possibly exponential size. Addressing this challenge, we propose an approach that
derives a performance-influence model for a given configurable system, describing
all relevant influences of configuration options and their interactions. Our approach
combines machine-learning and sampling heuristics in a novel way. It improves over
standard techniques in that it (1) represents influences of options and their interactions
explicitly (which eases debugging), (2) smoothly integrates binary and numeric configuration
options for the first time, (3) incorporates domain knowledge, if available (which
eases learning and increases accuracy), (4) considers complex constraints among options,
and (5) systematically reduces the solution space to a tractable size. A series of
experiments demonstrates the feasibility of our approach in terms of the accuracy
of the models learned as well as the accuracy of the performance predictions one can
make with them. },
booktitle = {Proceedings of the 2015 10th Joint Meeting on Foundations of Software Engineering},
pages = {284–294},
numpages = {11},
keywords = {machine learning, Performance-influence models, sampling},
location = {Bergamo, Italy},
series = {ESEC/FSE 2015}
}

Q1 : 
Yes, seven systems, including lrzip, x264, etc.

Q2 :
No, the experimental protocol is really impressive, but the measurements were done on one benchmark per system.
Below few examples to justify our answer:
x264 - "As benchmark, we measured the time needed to en-
code the Sintel trailer (734 MB) using on an Intel Core2
Q6600 with 4GB RAM (Ubuntu 14.04)" -> only one input video
JavaGC - "For measurement, we executed the Da-
Capo benchmark suite on a computing cluster consisting
of 16 nodes each equipped with an Intel Xeon E5-2690 Ivy
Bridge having 10 cores and 64 GB RAM (Ubuntu 14.04)." -> only one suite, without detailing each element
SaC - "As benchmark, we compile and execute an
n-body simulation shipped with the compiler, measuring
the execution time of the simulation at different optimiza-
tion levels." -> only one simulation/script

Q3 : 
No

Q4 : 
No, Q2 = No implies Q4 = No for this paper


### Paper 18

Title : 
Empirical Comparison of Regression Methods for Variability-Aware Performance Prediction

Bibtex :
@inproceedings{10.1145/2791060.2791069,
author = {Valov, Pavel and Guo, Jianmei and Czarnecki, Krzysztof},
title = {Empirical Comparison of Regression Methods for Variability-Aware Performance Prediction},
year = {2015},
isbn = {9781450336130},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/2791060.2791069},
doi = {10.1145/2791060.2791069},
abstract = {Product line engineering derives product variants by selecting features. Understanding
the correlation between feature selection and performance is important for stakeholders
to acquire a desirable product variant. We infer such a correlation using four regression
methods based on small samples of measured configurations, without additional effort
to detect feature interactions. We conduct experiments on six real-world case studies
to evaluate the prediction accuracy of the regression methods. A key finding in our
empirical study is that one regression method, called Bagging, is identified as the
best to make accurate and robust predictions for the studied systems.},
booktitle = {Proceedings of the 19th International Conference on Software Product Line},
pages = {186–190},
numpages = {5},
location = {Nashville, Tennessee},
series = {SPLC '15}
}

Q1 : 
Yes, 6 systems, including x264, SQLite, llvm, Apache

Q2 :
No, same data as for other SPLConqueror publications

Q3 : 
No

Q4 : 
No


### Paper 19

Title : 
Performance Prediction of Configurable Software Systems by Fourier Learning

Bibtex :
@inproceedings{10.1109/ASE.2015.15,
author = {Zhang, Yi and Guo, Jianmei and Blais, Eric and Czarnecki, Krzysztof},
title = {Performance Prediction of Configurable Software Systems by Fourier Learning},
year = {2015},
isbn = {9781509000241},
publisher = {IEEE Press},
url = {https://doi.org/10.1109/ASE.2015.15},
doi = {10.1109/ASE.2015.15},
abstract = {Understanding how performance varies across a large number of variants of a configurable
software system is important for helping stakeholders to choose a desirable variant.
Given a software system with n optional features, measuring all its 2n possible configurations
to determine their performances is usually infeasible. Thus, various techniques have
been proposed to predict software performances based on a small sample of measured
configurations. We propose a novel algorithm based on Fourier transform that is able
to make predictions of any configurable software system with theoretical guarantees
of accuracy and confidence level specified by the user, while using minimum number
of samples up to a constant factor. Empirical results on the case studies constructed
from real-world configurable systems demonstrate the effectiveness of our algorithm.},
booktitle = {Proceedings of the 30th IEEE/ACM International Conference on Automated Software Engineering},
pages = {365–373},
numpages = {9},
location = {Lincoln, Nebraska},
series = {ASE '15}
}

Q1 : 
Yes, 5 systems, including Apache, x264, llvm

Q2 :
No, same data as refs 6 and 13 using measurements with one input per system

Q3 : 
Yes. Justification:
"Since our approach is a black box method that operates on a
high level of abstraction, more software specific concerns such
as varying workload and multi-user scenarios might pose more
unexpected threats to our model. However one might be able
to see how these variations can be feasibly incorporated into
the modelling of features or performance objectives via some
transformation such as the ones outlined above, and hence
assume these threats are minimal."

Q4 : 
No

### Paper 20

Title : 
On the Relation of External and Internal Feature Interactions: A Case Study

Bibtex :
@misc{kolesnikov2018relation,
      title={On the Relation of External and Internal Feature Interactions: A Case Study}, 
      author={Sergiy Kolesnikov and Norbert Siegmund and Christian Kästner and Sven Apel},
      year={2018},
      eprint={1712.07440},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}

Q1 : 
Yes, including SQLite

Q2 : 
"The performance measurements were done using a standard benchmark"
-> No

Q3 : 
No mention of input sensitivity, no

Q4 : 
No 

### Paper 21

Title : 
Products Go Green: Worst-Case Energy Consumption in Software Product Lines

Bibtex :
@inproceedings{10.1145/3106195.3106214,
author = {Couto, Marco and Borba, Paulo and Cunha, J\'{a}come and Fernandes, Jo\~{a}o Paulo and Pereira, Rui and Saraiva, Jo\~{a}o},
title = {Products Go Green: Worst-Case Energy Consumption in Software Product Lines},
year = {2017},
isbn = {9781450352215},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3106195.3106214},
doi = {10.1145/3106195.3106214},
abstract = {The optimization of software to be (more) energy efficient is becoming a major concern
for the software industry. Although several techniques have been presented to measure
energy consumption for software, none has addressed software product lines (SPLs).
Thus, to measure energy consumption of a SPL, the products must be generated and measured
individually, which is too costly.In this paper, we present a technique and a prototype
tool to statically estimate the worst case energy consumption for SPL. The goal is
to provide developers with techniques and tools to reason about the energy consumption
of all products in a SPL, without having to produce, run and measure the energy in
all of them.Our technique combines static program analysis techniques and worst case
execution time prediction with energy consumption analysis. This technique analyzes
all products in a feature-sensitive manner, that is, a feature used in several products
is analyzed only once, while the energy consumption is estimated once per product.We
implemented our technique in a tool called Serapis. We did a preliminary evaluation
using a product line for image processing implemented in C. Our experiments considered
7 products from such line and our initial results show that the tool was able to estimate
the worst-case energy consumption with a mean error percentage of 9.4% and standard
deviation of 6.2% when compared with the energy measured when running the products.},
booktitle = {Proceedings of the 21st International Systems and Software Product Line Conference - Volume A},
pages = {84–93},
numpages = {10},
location = {Sevilla, Spain},
series = {SPLC '17}
}

Q1 : 
Yes, the products (ie the disparity tool) takes two images as inputs 

Q2 : 
Not sure, but seems to be a No
Here is a possible justification, seciton 5.2:
"This was performed by our dynamic energy mea-
suring framework, which is responsible for executing the products
with the same input and measure the energy consumption"

Q3 : 
Yes.
In related work, when the authors quote ref 17 
"Studies have shown that the energy consumption of a software
system can be signicantly inuenced by a lot of factors, such as
different design patterns [16], data structures [17], and refactor-
ings [20]."
We can consider it is a mention to input sensitivity to say that the energy consumption of a software system can be significantly influenced by data structures.

Q4 : 
No

### Paper 22

Title : 
Automatic Database Management System Tuning Through Large-Scale Machine Learning

Bibtex :
@inproceedings{10.1145/3035918.3064029,
author = {Van Aken, Dana and Pavlo, Andrew and Gordon, Geoffrey J. and Zhang, Bohan},
title = {Automatic Database Management System Tuning Through Large-Scale Machine Learning},
year = {2017},
isbn = {9781450341974},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3035918.3064029},
doi = {10.1145/3035918.3064029},
abstract = {Database management system (DBMS) configuration tuning is an essential aspect of any
data-intensive application effort. But this is historically a difficult task because
DBMSs have hundreds of configuration "knobs" that control everything in the system,
such as the amount of memory to use for caches and how often data is written to storage.
The problem with these knobs is that they are not standardized (i.e., two DBMSs use
a different name for the same knob), not independent (i.e., changing one knob can
impact others), and not universal (i.e., what works for one application may be sub-optimal
for another). Worse, information about the effects of the knobs typically comes only
from (expensive) experience.To overcome these challenges, we present an automated
approach that leverages past experience and collects new information to tune DBMS
configurations: we use a combination of supervised and unsupervised machine learning
methods to (1) select the most impactful knobs, (2) map unseen database workloads
to previous workloads from which we can transfer experience, and (3) recommend knob
settings. We implemented our techniques in a new tool called OtterTune and tested
it on two DBMSs. Our evaluation shows that OtterTune recommends configurations that
are as good as or better than ones generated by existing tools or a human expert.},
booktitle = {Proceedings of the 2017 ACM International Conference on Management of Data},
pages = {1009–1024},
numpages = {16},
keywords = {autonomic computing, database tuning, database management systems, machine learning},
location = {Chicago, Illinois, USA},
series = {SIGMOD '17}
}

Q1 : 
Yes, 3 DBMS including PostG, mySQL

Q2 : 
Yes

Q3 : 
Yes, the purpose of this paper

Q4 : 
"It then
creates models from this data that allow it to (1) select the most
impactful knobs, (2) map previously unseen database workloads to
known workloads, and (3) recommend knob settings. We start with
discussing how to identify which of the metrics gathered by the tun-
ing tool best characterize an application’s workload."

YES, definitely a candidate solution to generalize to other software systems dealing with input sensitvity!
See also figure 3, interesting first step to characterize the workload.
Looks like the paper of Ding et al. on Petabricks, a first step to classify the input of input, and then a second step to use these input properties to predict accurately the configs.


### Paper 23

Title : 
Distance-Based Sampling of Software Configuration Spaces

Bibtex :
@inproceedings{10.1109/ICSE.2019.00112,
author = {Kaltenecker, Christian and Grebhahn, Alexander and Siegmund, Norbert and Guo, Jianmei and Apel, Sven},
title = {Distance-Based Sampling of Software Configuration Spaces},
year = {2019},
publisher = {IEEE Press},
url = {https://doi.org/10.1109/ICSE.2019.00112},
doi = {10.1109/ICSE.2019.00112},
abstract = {Configurable software systems provide a multitude of configuration options to adjust
and optimize their functional and non-functional properties. For instance, to find
the fastest configuration for a given setting, a brute-force strategy measures the
performance of all configurations, which is typically intractable. Addressing this
challenge, state-of-the-art strategies rely on machine learning, analyzing only a
few configurations (i.e., a sample set) to predict the performance of other configurations.
However, to obtain accurate performance predictions, a representative sample set of
configurations is required. Addressing this task, different sampling strategies have
been proposed, which come with different advantages (e.g., covering the configuration
space systematically) and disadvantages (e.g., the need to enumerate all configurations).
In our experiments, we found that most sampling strategies do not achieve a good coverage
of the configuration space with respect to covering relevant performance values. That
is, they miss important configurations with distinct performance behavior. Based on
this observation, we devise a new sampling strategy, called distance-based sampling,
that is based on a distance metric and a probability distribution to spread the configurations
of the sample set according to a given probability distribution across the configuration
space. This way, we cover different kinds of interactions among configuration options
in the sample set. To demonstrate the merits of distance-based sampling, we compare
it to state-of-the-art sampling strategies, such as t-wise sampling, on 10 real-world
configurable software systems. Our results show that distance-based sampling leads
to more accurate performance models for medium to large sample sets.},
booktitle = {Proceedings of the 41st International Conference on Software Engineering},
pages = {1084–1094},
numpages = {11},
location = {Montreal, Quebec, Canada},
series = {ICSE '19}
}

Q1 : 
Yes, 10 real-world configurable software, including 7zip and x264

Q2 : 
No, even if there are lots of measurements, the data for each systems https://github.com/se-passau/Distance-Based_Data/tree/master/SupplementaryWebsite/MeasuredPerformanceValues
only contains one performance
They consider well-known benchmarks, but they did not provide the details e.g. for 7zip

"We used version 9.20 of 7-ZIP and measured the compression time of the Canterbury corpus6 on an Intel Xeon E5-2690 and 64 GB RAM (Ubuntu 16.04)."
Yes, but the detail for each file of the Canterbury corpus is not available in the repository, you just measured the time to compress the whole content of the corpus, i.e. as a unique folder. 
And it is the same for other software systems; Sintel for 264, the gemm program for Polly, etc.
It means that you will encounter the input sensitivity presented in this submission: Sintel is just one video, representing one profile of encoding.

So, for the second question of this paper, we are forced to answer No, even if the protocol is well-designed and really impressive in terms of measurement efforts.

Q3 : 
No

Q4 : 
No

### Paper 24

Title : 
Learning to Sample: Exploiting Similarities across Environments to Learn Performance Models for Configurable Systems

Bibtex :
@inproceedings{10.1145/3236024.3236074,
author = {Jamshidi, Pooyan and Velez, Miguel and K\"{a}stner, Christian and Siegmund, Norbert},
title = {Learning to Sample: Exploiting Similarities across Environments to Learn Performance Models for Configurable Systems},
year = {2018},
isbn = {9781450355735},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3236024.3236074},
doi = {10.1145/3236024.3236074},
abstract = {Most software systems provide options that allow users to tailor the system in terms
of functionality and qualities. The increased flexibility raises challenges for understanding
the configuration space and the effects of options and their interactions on performance
and other non-functional properties. To identify how options and interactions affect
the performance of a system, several sampling and learning strategies have been recently
proposed. However, existing approaches usually assume a fixed environment (hardware,
workload, software release) such that learning has to be repeated once the environment
changes. Repeating learning and measurement for each environment is expensive and
often practically infeasible. Instead, we pursue a strategy that transfers knowledge
across environments but sidesteps heavyweight and expensive transfer-learning strategies.
Based on empirical insights about common relationships regarding (i) influential options,
(ii) their interactions, and (iii) their performance distributions, our approach,
L2S (Learning to Sample), selects better samples in the target environment based on
information from the source environment. It progressively shrinks and adaptively concentrates
on interesting regions of the configuration space. With both synthetic benchmarks
and several real systems, we demonstrate that L2S outperforms state of the art performance
learning and transfer-learning approaches in terms of measurement effort and learning
accuracy.},
booktitle = {Proceedings of the 2018 26th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
pages = {71–82},
numpages = {12},
keywords = {Software performance, configurable systems, transfer learning},
location = {Lake Buena Vista, FL, USA},
series = {ESEC/FSE 2018}
}

Q1 : 
Yes, 4 software systems, including Apache Storm

Q2 : 
Yes, e.g.
"We selected 11 configuration options and measured through-put as response on standard benchmarks (SOL, WordCount, and RollingCount)."

Q3 : 
Yes,
"We used two standard datasets as workload to train models and measure
training time as the response variable. We varied both hardware
(expected easy environment change) and workload (hard)."

Q4 : 
Yes, the method proposed in this paper adapts a performance model trained on a source environment (i.e. with a given input) and transfer it to a target environment (i.e. with another input).
Indeed a good candidate solution to test on our data.


### Paper 25

Title : 
An Uncertainty-Aware Approach to Optimal Configuration of Stream Processing Systems

Bibtex :
@misc{jamshidi2016uncertaintyaware,
      title={An Uncertainty-Aware Approach to Optimal Configuration of Stream Processing Systems}, 
      author={Pooyan Jamshidi and Giuliano Casale},
      year={2016},
      eprint={1606.06543},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}

Q1 : 
Yes, Apache Storm

Q2 : 
Yes. Justification: 
"In this section, we evaluate BO4CO using 3 different Storm
benchmarks: (i) WordCount, (ii) RollingSort, (iii) SOL.
RollingSort implements a common pattern in real-time data
analysis that performs rolling counts of incoming messages."

Q3 : 
Yes.

"First, the performance
difference between the best and worst settings is substantial,
65%, and with more intense workloads we have observed
differences in latency as large as 99%, see Table V. "

This comment + table V showing the differences obtained with different settings depending on the workload -> basically our RQ3 in this paper

Q4 : 
No

### Paper 26

Title : 
Improved prediction of non-functional properties in software product lines with domain context

Bibtex :
@article{lillack2013improved,
  title={Improved prediction of non-functional properties in software product lines with domain context},
  author={Lillack, Max and M{\"u}ller, Johannes and Eisenecker, Ulrich W},
  journal={Software Engineering 2013},
  year={2013},
  publisher={Gesellschaft f{\"u}r Informatik eV}
}

Q1 : 
Yes, eg Compressor SPL process input data

Q2 : 
Yes, the  approach proposed in this paper generates input data, on which they apply performance models.

Q3 : 
Yes, this is the point of the paper actually, see section 4 e.g.
"Our hypothesis is: considering input data in the measurement can improve the prediction of non-functional properties."
The underlying assumption is input sensitivity. 
+
"Despite the fact that our case study is based on an artificial SPL we can see that input
data can indeed have an effect on non-functional properties."
Yes

Q4 : 
Sound approach to test and reproduce on other domains than data compression, and maybe to combine with real data (i.e. in the approach, instead of generating random data, generate data based on the distribution really observed with real cases)
Yes


### Paper 27

Title : 
e-PAL: An Active Learning Approach to the Multi-Objective Optimization Problem

Bibtex :
@article{JMLR:v17:15-047,
  author  = {Marcela Zuluaga and Andreas Krause and Markus P{{{\"u}}}schel},
  title   = {e-PAL: An Active Learning Approach to the Multi-Objective Optimization Problem},
  journal = {Journal of Machine Learning Research},
  year    = {2016},
  volume  = {17},
  number  = {104},
  pages   = {1-32},
  url     = {http://jmlr.org/papers/v17/15-047.html}
}

Q1 : 
Yes, 3 real-world cases, including llvm

Q2 : 
"The objectives are performance and memory
footprint for a given suite of software programs when compiled with these settings."
+
https://github.com/FlashRepo/epsilon-PAL/tree/master/results_llvm
Yes

Q3 : 
No

Q4 : 
No

### Paper 28

Title : 
Towards Learning-Aided Configuration in 3D Printing: Feasibility Study and Application to Defect Prediction

Bibtex :
@inproceedings{10.1145/3302333.3302338,
author = {Amand, Benoit and Cordy, Maxime and Heymans, Patrick and Acher, Mathieu and Temple, Paul and J\'{e}z\'{e}quel, Jean-Marc},
title = {Towards Learning-Aided Configuration in 3D Printing: Feasibility Study and Application to Defect Prediction},
year = {2019},
isbn = {9781450366489},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3302333.3302338},
doi = {10.1145/3302333.3302338},
abstract = {Configurators rely on logical constraints over parameters to aid users and determine
the validity of a configuration. However, for some domains, capturing such configuration
knowledge is hard, if not infeasible. This is the case in the 3D printing industry,
where parametric 3D object models contain the list of parameters and their value domains,
but no explicit constraints. This calls for a complementary approach that learns what
configurations are valid based on previous experiences. In this paper, we report on
preliminary experiments showing the capability of state-of-the-art classification
algorithms to assist the configuration process. While machine learning holds its promises
when it comes to evaluation scores, an in-depth analysis reveals the opportunity to
combine the classifiers with constraint solvers.},
booktitle = {Proceedings of the 13th International Workshop on Variability Modelling of Software-Intensive Systems},
articleno = {7},
numpages = {9},
keywords = {Sampling, 3D printing, Machine Learning, Configuration},
location = {Leuven, Belgium},
series = {VAMOS '19}
}

Q1 : 
Yes, Thingiverse process 3D models and prints them

Q2 : 
"In the end, 31 models remained whose topology are presented in
Figure 3."
Yes

Q3 : 
"The first threat is the set of 3D models we selected
to analyze. In particular, we discarded models requiring substantial
computation resources (due to a large number of configurations or
to a high analysis time), as our goal was only to gain first insights.
However, this made us ignore models with large configuration
space, which are likely to be more challenging for the classifiers."
Yes

Q4 :
No. Justification here:
"The first part is a quantita-
tive evaluation of the performance of each classifier trained and
evaluated on each model separately"

### Paper 29

Title : 

Bibtex :

Q1 : 
Yes, spark & hadoop

Q2 : 
Yes, multiple benchmarks to test them

" 1) TPC-DS [7] [...]
2) TPC-H [8] [...]
3) 3) TeraSort [29] [...]
4) 4) The SparkReg [4] [...]
5) SparkKm is another SparkML benchmark"

Q3 : 
"CherryPick depends on representative workloads. Thus,
one concern is CherryPick’s sensitivity to the variation
of input workloads."
Yes

Q4 : 

Tempted to say yes, because Cherrypick is almost there
"CherryPick detects the need
to recompute the cloud configuration when it finds large
gaps between estimated performance and real perfor-
mance under the current configuration"

But since Cherrypick depends on representative workloads chosen by humans to work, it is not yet an automated process.

"Picking a representative workload for non-
recurring jobs hard, and for now, CherryPick relies on
human intuitions. An automatic way to select represen-
tative workload is an interesting avenue for future work"

That is why we choose "No" (but it can be discussed)


### Paper 30

Title : 

Bibtex :

Q1 : 

Q2 : 

Q3 : 

Q4 : 

### Paper 31

Title : 

Bibtex :

Q1 : 

Q2 : 

Q3 : 

Q4 : 

### Paper 32

Title : 

Bibtex :

Q1 : 

Q2 : 

Q3 : 

Q4 : 

### Paper 33

Title : 

Bibtex :

Q1 : 

Q2 : 

Q3 : 

Q4 : 

### Paper 34

Title : 

Bibtex :

Q1 : 

Q2 : 

Q3 : 

Q4 : 

### Paper 35

Title : 

Bibtex :

Q1 : 

Q2 : 

Q3 : 

Q4 : 

### Paper 36

Title : 

Bibtex :

Q1 : 

Q2 : 

Q3 : 

Q4 : 

### Paper 37

Title : 

Bibtex :

Q1 : 

Q2 : 

Q3 : 

Q4 : 

### Paper 38

Title : 

Bibtex :

Q1 : 

Q2 : 

Q3 : 

Q4 : 

### Paper 39

Title : 

Bibtex :

Q1 : 

Q2 : 

Q3 : 

Q4 : 

### Paper 40

Title : 

Bibtex :

Q1 : 

Q2 : 

Q3 : 

Q4 : 
