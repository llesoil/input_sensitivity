
# Input sensitivity in research

This part addresses the following question :

## How do state-of-the-art papers address input sensitivity?

To do so, we gather research papers (see the protocol of the related submission for details).

We read each of them carefully and  answer  four  different  questions:

### Q-A.  Do  the  software systems process input data?

If most of research papers do not study configurable systems processing input data, the impact of input sensitivity in research would be relatively low. The idea of this research question is to estimate which proportion of the performance models could be affected by input sensitivity. 

### Q-B. Does the experimental protocol consider several inputs?

Then, we check if the research papers include several inputs in the study.  If  not,  it  would  suggest  that  the  performance  model only captures a partial truth, and might not generalize for other inputs fed to the software system. 

### Q-C. Is the problem of input sensitivity mentioned e.g., in threat? 

This question aims to state whether researchers are aware of the input sensitivity threat, and which proportion of the papers mention it as a potential threat to validity. 

### Q-D. Does the paper propose a solution to generalize the performance model across inputs? 

Finally, we check if the paper proposes a solution solving input sensitivity i.e., if they are able to train a performance model that is general and robust enough to predict a near-optimal configuration for any input. 

The results were obtained by one author and validated by two authors of this paper. 

Feel free to contact us if you disagree with the results, or if any comment seems unfair to you!

## RESULTS

The rest of this document details the results for each paper and justifies the answers to Q-A, Q-B, Q-C and Q-D. 

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

Q-A : 
Yes, 10 configurable systems, including x264, SQLite, llvm, etc.

Q-B : 
"We measured their performance using standard benchmarks from the respective application domain." -> but looking at the material  https://github.com/jmguo/DECART/blob/master/multipleMetrics/results/CART_x264_Metric1and2_Details_crossvalidation_gridsearch.csv
seems to be only one input per system -> No

Q-C : 
"We measured their performance using standard benchmarks from the respective application domain." in threats, but no mention of a potential lack of generalization -> No

Q-D : 
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

Q-A : 
Yes, Cassandra, nosql database

Q-B : 
Yes, see Table 1 in the supplementary material here https://github.com/pooyanjamshidi/transferlearning/blob/master/online-appendix.pdf

Q-C : 
In the article, section Threats to validity "Moreover,
we used standard benchmarks so that we are confident in that
we have measured a realistic scenario." -> Yes

Q-D : 
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

Q-A : 
Yes, 4 configurable systems, including x264 and SQLite

Q-B : 
Yes, different workloads, see |W| in Table 1

Q-C : 
Yes, "We selected a diverse set of subject
systems and a large number of purposefully selected environ-
ment changes, but, as usual, one has to be careful when gen-
eralizing to other subject systems and environment changes." in threats section 

Q-D : 
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

Q-A : 
Yes, x264 and llvm at least

Q-B : 
In threats "We used ground-truth data of [30] which are
measurements of real systems."
Following the link in ref 30, http://fosd.de/SPLConqueror
it is yet unclear which data are used for this paper.
Assuming the data are those from the ICSE 2012 paper, there is only one input per system when downloading the supplementary material -> No

Q-C : 
No mention of input sensitivity, No

Q-D : 
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

Q-A : 
Yes, 10 real-world highly-configurable systems, including x264

Q-B : 
According to the paper section 3.3 "using standard benchmarks for the respective domain", but following https://www.se.cs.uni-saarland.de/projects/tradeoffs/
only one input per system ? -> No

Q-C : 
No mention of input sensitivity, No

Q-D : 
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

Q-A : 
Yes, 9 software systems including berkeley lrzip, etc.

Q-B : 

Depending on the software, yes or no, e.g.

No for SQLite
"For SQLite, we cannot measure all possible configurations inreasonable time. Hence, we sampled only 100 configurations tocompare prediction and actual values. We are aware that this evalu-ation leaves room for outliers and that measurement bias can causefalse interpretations [11]. Since we limit our attention to predictingperformance for a given workload, we did not vary benchmarks."

Yes for Apache Storm, it considers several inputs
"The experiment considers three benchmarks namely:
WordCount (wc) counts the number of occurences of thewords in a text file.
RollingSort (rs) implements a common pattern in real-timeanalysis that performs rolling counts of messages.
SOL (sol) is a network intensive topology, where the mes-sage is routed through an inter-worker network"

We choose to answer "Yes" for Q-B.

Q-C : 
No mention of input sensitivity, No

Q-D : 
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

Q-A : 
Yes, including x264, llvm, etc.

Q-B : 
Word Count and Rolling Sort executed with Stream Processing Systems
Not many inputs, but we can answer Yes

Q-C : 
"Hence,  there  is  noinherent  mechanism  in  FLASH which  would  adapt  itself based  on  the  change  in  workload.  This  non-stationary nature  of  the  problem  is  a  significant  assumption  and currently  not  addressed  in  this  paper."
-> Yes

Q-D : 
Same citation as Q-C, the paper is honest on the limitations of the approach.
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

Q-A :
Yes, Apache, Nginx, and combination of tools that takes an input (web content)

Q-B : 
Yes, e.g.
"Further, we use Machine Learning technique to pre-dict individual component and configuration energy usagewith varied workload"


Q-C : 
Yes, they even measure different performance values for different values, which is highly valuable in terms of input sensitivity.
"This gives the insight that, depending on the workload, different configurations are unlikely to lead to significant impact on CPU power consumption."
6.1.2 Experimental Results is really an interesting case that carry the same message as our paper.

Q-D : 
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

Q-A : 
No, the video generator do not consider any input data

Q-B : 
Q-A = No

Q-C : 
Q-A = No

Q-D : 
Q-A = No

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

Q-A : 
Yes, 10 systems, including x264, SQLite

Q-B : 
One input per system, see https://github.com/learningconstraints/ICSE-17/tree/master/datasets
following the link of http://learningconstraints.github.io/
-> No

Q-C : 
"A first difficulty is related to the development of procedures (oracles) for measuring software configurations in different contexts. It may be difficult to find the right data or to create the realistic contextual conditions"
-> Yes

Q-D : 
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

Q-A : 
Yes, including x264, xz and sqlite

Q-B : 
No, they just make the hardware vary.

Q-C : 
In future work, 
"We  suspect  that  variations in system workload might influence transferability of system’sperformance prediction model by distorting system’s performance distribution across different hardware environments."
-> Yes + plus an interesting idea to combine hardware and input data

Q-D : 
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

Q-A : 
No, the simulator do not consider any input data

Q-B : 
Q-A = No

Q-C : 
Q-A = No

Q-D : 
Q-A = No

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

Q-A : 
Yes, the software process different scripts of latex code

Q-B : 
Yes, different pdf codes are fed to the system

Q-C : 
No ?

Q-D : 
No, the model does not generalize

### Paper 14

Title : 
Cost-Efficient Sampling for Performance Prediction of Configurable Systems

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

Q-A : 
Yes, 6 systems, including x264, llvm, SQLite, etc.

Q-B : 
No, they focus on configuration options (they are mostly interested in varying the sample size) so it explains why

Q-C : 
No

Q-D : 
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

Q-A : 
No, the video generator do not consider any input data

Q-B : 
Q-A = No

Q-C : 
Q-A = No

Q-D : 
Q-A = No

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

Q-A : 
Yes, including x264, SQLite, Apache

Q-B : 
 -> SQLite
Only one input per system e.g. Sintel for x264, the test suite (i.e. without detailing each test) for llvm, etc.
-> No

Q-C : 
No

Q-D : 
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

Q-A : 
Yes, seven systems, including lrzip, x264, etc.

Q-B :
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

Q-C : 
No

Q-D : 
No, Q-B = No implies Q-D = No for this paper


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

Q-A : 
Yes, 6 systems, including x264, SQLite, llvm, Apache

Q-B :
No, same data as for other SPLConqueror publications

Q-C : 
No

Q-D : 
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

Q-A : 
Yes, 5 systems, including Apache, x264, llvm

Q-B :
No, same data as refs 6 and 13 using measurements with one input per system

Q-C : 
Yes. Justification:
"Since our approach is a black box method that operates on a
high level of abstraction, more software specific concerns such
as varying workload and multi-user scenarios might pose more
unexpected threats to our model. However one might be able
to see how these variations can be feasibly incorporated into
the modelling of features or performance objectives via some
transformation such as the ones outlined above, and hence
assume these threats are minimal."

Q-D : 
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

Q-A : 
Yes, including SQLite

Q-B : 
"The performance measurements were done using a standard benchmark"
-> No

Q-C : 
No mention of input sensitivity, no

Q-D : 
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

Q-A : 
Yes, the products (ie the disparity tool) takes two images as inputs 

Q-B : 
Not sure, but seems to be a No
Here is a possible justification, seciton 5.2:
"This was performed by our dynamic energy mea-
suring framework, which is responsible for executing the products
with the same input and measure the energy consumption"

Q-C : 
Yes.
In related work, when the authors quote ref 17 
"Studies have shown that the energy consumption of a software
system can be signicantly inuenced by a lot of factors, such as
different design patterns [16], data structures [17], and refactor-
ings [20]."
We can consider it is a mention to input sensitivity to say that the energy consumption of a software system can be significantly influenced by data structures.

Q-D : 
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

Q-A : 
Yes, 3 DBMS including PostG, mySQL

Q-B : 
Yes

Q-C : 
Yes, the purpose of this paper

Q-D : 
"It then
creates models from this data that allow it to (1) select the most
impactful knobs, (2) map previously unseen database workloads to
known workloads, and (3) recommend knob settings. We start with
discussing how to identify which of the metrics gathered by the tun-
ing tool best characterize an application’s workload."

YES, definitely a candidate solution to generalize to other software systems dealing with input sensitvity!
But only on a small domain.
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

Q-A : 
Yes, 10 real-world configurable software, including 7zip and x264

Q-B : 
No, even if there are lots of measurements, the data for each systems https://github.com/se-passau/Distance-Based_Data/tree/master/SupplementaryWebsite/MeasuredPerformanceValues
only contains one performance
They consider well-known benchmarks, but they did not provide the details e.g. for 7zip

"We used version 9.20 of 7-ZIP and measured the compression time of the Canterbury corpus6 on an Intel Xeon E5-2690 and 64 GB RAM (Ubuntu 16.04)."
Yes, but the detail for each file of the Canterbury corpus is not available in the repository, you just measured the time to compress the whole content of the corpus, i.e. as a unique folder. 
And it is the same for other software systems; Sintel for 264, the gemm program for Polly, etc.
It means that you will encounter the input sensitivity presented in this submission: Sintel is just one video, representing one profile of encoding.

So, for the second question of this paper, we are forced to answer No, even if the protocol is well-designed and really impressive in terms of measurement efforts.

Q-C : 
No

Q-D : 
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

Q-A : 
Yes, 4 software systems, including Apache Storm

Q-B : 
Yes, e.g.
"We selected 11 configuration options and measured through-put as response on standard benchmarks (SOL, WordCount, and RollingCount)."

Q-C : 
Yes,
"We used two standard datasets as workload to train models and measure
training time as the response variable. We varied both hardware
(expected easy environment change) and workload (hard)."

Q-D : 
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

Q-A : 
Yes, Apache Storm

Q-B : 
Yes. Justification: 
"In this section, we evaluate BO4CO using 3 different Storm
benchmarks: (i) WordCount, (ii) RollingSort, (iii) SOL.
RollingSort implements a common pattern in real-time data
analysis that performs rolling counts of incoming messages."

Q-C : 
Yes.

"First, the performance
difference between the best and worst settings is substantial,
65%, and with more intense workloads we have observed
differences in latency as large as 99%, see Table V. "

This comment + table V showing the differences obtained with different settings depending on the workload -> basically our RQ-C in this paper

Q-D : 
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

Q-A : 
Yes, eg Compressor SPL process input data

Q-B : 
Yes, the  approach proposed in this paper generates input data, on which they apply performance models.

Q-C : 
Yes, this is the point of the paper actually, see section 4 e.g.
"Our hypothesis is: considering input data in the measurement can improve the prediction of non-functional properties."
The underlying assumption is input sensitivity. 
+
"Despite the fact that our case study is based on an artificial SPL we can see that input
data can indeed have an effect on non-functional properties."
Yes

Q-D : 
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

Q-A : 
Yes, 3 real-world cases, including llvm

Q-B : 
"The objectives are performance and memory
footprint for a given suite of software programs when compiled with these settings."
+
https://github.com/FlashRepo/epsilon-PAL/tree/master/results_llvm
Yes

Q-C : 
No

Q-D : 
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

Q-A : 
Yes, Thingiverse process 3D models and prints them

Q-B : 
"In the end, 31 models remained whose topology are presented in
Figure 3."
Yes

Q-C : 
"The first threat is the set of 3D models we selected
to analyze. In particular, we discarded models requiring substantial
computation resources (due to a large number of configurations or
to a high analysis time), as our goal was only to gain first insights.
However, this made us ignore models with large configuration
space, which are likely to be more challenging for the classifiers."
Yes

Q-D :
No. Justification here:
"The first part is a quantita-
tive evaluation of the performance of each classifier trained and
evaluated on each model separately"

### Paper 29

Title : 
Cherrypick: adaptively unearthing the best cloud configurations for big data analytics

Bibtex :
@inproceedings{10.5555/3154630.3154669,
author = {Alipourfard, Omid and Liu, Hongqiang Harry and Chen, Jianshu and Venkataraman, Shivaram and Yu, Minlan and Zhang, Ming},
title = {Cherrypick: Adaptively Unearthing the Best Cloud Configurations for Big Data Analytics},
year = {2017},
isbn = {9781931971379},
publisher = {USENIX Association},
address = {USA},
abstract = {Picking the right cloud configuration for recurring big data analytics jobs running
in clouds is hard, because there can be tens of possible VM instance types and even
more cluster sizes to pick from. Choosing poorly can significantly degrade performance
and increase the cost to run a job by 2-3x on average, and as much as 12x in the worst-case.
However, it is challenging to automatically identify the best configuration for a
broad spectrum of applications and cloud configurations with low search cost. CherryPick
is a system that leverages Bayesian Optimization to build performance models for various
applications, and the models are just accurate enough to distinguish the best or close-to-the-best
configuration from the rest with only a few test runs. Our experiments on five analytic
applications in AWS EC2 show that CherryPick has a 45-90% chance to find optimal configurations,
otherwise near-optimal, saving up to 75% search cost compared to existing solutions.},
booktitle = {Proceedings of the 14th USENIX Conference on Networked Systems Design and Implementation},
pages = {469–482},
numpages = {14},
location = {Boston, MA, USA},
series = {NSDI'17}
}

Q-A : 
Yes, spark & hadoop

Q-B : 
Yes, multiple benchmarks to test them

" 1) TPC-DS [7] [...]
2) TPC-H [8] [...]
3) 3) TeraSort [29] [...]
4) 4) The SparkReg [4] [...]
5) SparkKm is another SparkML benchmark"

Q-C : 
"CherryPick depends on representative workloads. Thus,
one concern is CherryPick’s sensitivity to the variation
of input workloads."
Yes

Q-D : 

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
Personalized Decision-Strategy based Web Service Selection using a Learning-to-Rank Algorithm

Bibtex :
@ARTICLE{6981951,
author={Saleem, Muhammad Suleman and Ding, Chen and Liu, Xumin and Chi, Chi-Hung},
journal={IEEE Transactions on Services Computing}, 
title={Personalized Decision-Strategy based Web Service Selection using a Learning-to-Rank Algorithm}, 
year={2015},
volume={8},
number={5},
pages={727-739},
doi={10.1109/TSC.2014.2377724}
}

Note to read before reading the following justifications. We kept this paper because, in a way, the following paper is related to input sensitivity.
Here, a user is an "input data"; he has his/her own decision strategy, and based on the decisions he makes, we can predict the best choice for her/him.
When transposing this to configurable systems, it could be an algorithm predicting the best configuration (aka the best choice) for a given input data.

Q-A : 
See the previous justification, yes.

Q-B : 
We do not know how many users there are in the generated data, but according to "In the generated dataset, each user submitted 100
queries, and each query has requirements on 1-7 QoS prop-
erties" we can say Yes.

Q-C : 
No

Q-D : 
No

### Paper 31

Title : 
A Mathematical Model of Performance-Relevant Feature Interactions

Bibtex :
@inproceedings{10.1145/2934466.2934469,
author = {Zhang, Yi and Guo, Jianmei and Blais, Eric and Czarnecki, Krzysztof and Yu, Huiqun},
title = {A Mathematical Model of Performance-Relevant Feature Interactions},
year = {2016},
isbn = {9781450340502},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/2934466.2934469},
doi = {10.1145/2934466.2934469},
abstract = {Modern software systems have grown significantly in their size and complexity, therefore
understanding how software systems behave when there are many configuration options,
also called features, is no longer a trivial task. This is primarily due to the potentially
complex interactions among the features. In this paper, we propose a novel mathematical
model for performance-relevant, or quantitative in general, feature interactions,
based on the theory of Boolean functions. Moreover, we provide two algorithms for
detecting all such interactions with little measurement effort and potentially guaranteed
accuracy and confidence level. Empirical results on real-world configurable systems
demonstrated the feasibility and effectiveness of our approach.},
booktitle = {Proceedings of the 20th International Systems and Software Product Line Conference},
pages = {25–34},
numpages = {10},
keywords = {fourier transform, feature interactions, performance, boolean functions},
location = {Beijing, China},
series = {SPLC '16}
}

Q-A : 
Yes, 4 four systems, including Apache, llvm and x264

Q-B : 
No, same dataset as SPLConqueror (see previous justifications)

Q-C : 
No, neither detailed in the evaluation nor in threats

Q-D : 
No, Q-B = No implies Q-D = No for this paper


### Paper 32

Title : 
Automated Search for Configurations of Convolutional Neural Network Architectures

Bibtex :
@inproceedings{10.1145/3336294.3336306,
author = {Ghamizi, Salah and Cordy, Maxime and Papadakis, Mike and Traon, Yves Le},
title = {Automated Search for Configurations of Convolutional Neural Network Architectures},
year = {2019},
isbn = {9781450371384},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3336294.3336306},
doi = {10.1145/3336294.3336306},
abstract = {Convolutional Neural Networks (CNNs) are intensively used to solve a wide variety
of complex problems. Although powerful, such systems require manual configuration
and tuning. To this end, we view CNNs as configurable systems and propose an end-to-end
framework that allows the configuration, evaluation and automated search for CNN architectures.
Therefore, our contribution is threefold. First, we model the variability of CNN architectures
with a Feature Model (FM) that generalizes over existing architectures. Each valid
configuration of the FM corresponds to a valid CNN model that can be built and trained.
Second, we implement, on top of Tensorflow, an automated procedure to deploy, train
and evaluate the performance of a configured model. Third, we propose a method to
search for configurations and demonstrate that it leads to good CNN models. We evaluate
our method by applying it on image classification tasks (MNIST, CIFAR-10) and show
that, with limited amount of computation and training, our method can identify high-performing
architectures (with high accuracy). We also demonstrate that we outperform existing
state-of-the-art architectures handcrafted by ML researchers. Our FM and framework
have been released to support replication and future research.},
booktitle = {Proceedings of the 23rd International Systems and Software Product Line Conference - Volume A},
pages = {119–130},
numpages = {12},
keywords = {NAS, configuration search, feature model, neural architecture search, AutoML},
location = {Paris, France},
series = {SPLC '19}
}

Q-A : 
Yes, the neural network are processing images

Q-B : 
Yes, they used MNIST and CIFAR-10

Q-C : 
Yes, justification
"As we did not
experiment with datasets involving a large variety of many or few classes and/or bigger images, it could be the case
that our results are coincidental and do not generalize to other cases. Though, our framework manages to identify
architectures leading to a large range of results indicating its ability to specialize to specific cases"

Q-D : 
No, it is a per-dataset architecture prediction.


### Paper 33

Title : 
Performance‐influence models of multigrid methods: A case study on triangular grids

Bibtex :
@article{Grebhahn2017PerformanceinfluenceMO,
  title={Performance‐influence models of multigrid methods: A case study on triangular grids},
  author={A. Grebhahn and C. Rodrigo and Norbert Siegmund and F. Gaspar and S. Apel},
  journal={Concurrency and Computation: Practice and Experience},
  year={2017},
  volume={29}
}

Q-A : 
Don't see any input. The ground-truth is used as validation (so not as input), and the Poisson's equation stays the same.
No ?

Q-B : 
Q-A = No

Q-C : 
Q-A = No

Q-D : 
Q-A = No


### Paper 34

Title : 
AutoConfig: Automatic Configuration Tuning for Distributed Message Systems

Bibtex :
@inproceedings{10.1145/3238147.3238175,
author = {Bao, Liang and Liu, Xin and Xu, Ziheng and Fang, Baoyin},
title = {AutoConfig: Automatic Configuration Tuning for Distributed Message Systems},
year = {2018},
isbn = {9781450359375},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3238147.3238175},
doi = {10.1145/3238147.3238175},
abstract = {Distributed message systems (DMSs) serve as the communication backbone for many real-time
streaming data processing applications. To support the vast diversity of such applications,
DMSs provide a large number of parameters to configure. However, It overwhelms for
most users to configure these parameters well for better performance. Although many
automatic configuration approaches have been proposed to address this issue, critical
challenges still remain: 1) to train a better and robust performance prediction model
using a limited number of samples, and 2) to search for a high-dimensional parameter
space efficiently within a time constraint. In this paper, we propose AutoConfig --
an automatic configuration system that can optimize producer-side throughput on DMSs.
AutoConfig constructs a novel comparison-based model (CBM) that is more robust that
the prediction-based model (PBM) used by previous learning-based approaches. Furthermore,
AutoConfig uses a weighted Latin hypercube sampling (wLHS) approach to select a set
of samples that can provide a better coverage over the high-dimensional parameter
space. wLHS allows AutoConfig to search for more promising configurations using the
trained CBM. We have implemented AutoConfig on the Kafka platform, and evaluated it
using eight different testing scenarios deployed on a public cloud. Experimental results
show that our CBM can obtain better results than that of PBM under the same random
forests based model. Furthermore, AutoConfig outperforms default configurations by
215.40% on average, and five state-of-the-art configuration algorithms by 7.21%-64.56%.},
booktitle = {Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering},
pages = {29–40},
numpages = {12},
keywords = {comparison-based model, automatic configuration tuning, distributed message system, weighted Latin hypercube sampling},
location = {Montpellier, France},
series = {ASE 2018}
}

Q-A : 
Yes, Kafka, processing messages

Q-B : 
Yes, various types of messages, see Table II and
"Benchmark. Based on the previous Kafka testing experiences
[25, 40], we design eight testing scenarios with the combination-
s of different numbers of producers, message sizes, and message
acknowledgement modes. Table 2 lists these testing setups."

Q-C : 
No

Q-D : 
No

### Paper 35

Title : 
Variability-aware performance prediction: A statistical learning approach

Bibtex :
@INPROCEEDINGS{6693089,
author={Guo, Jianmei and Czarnecki, Krzysztof and Apel, Sven and Siegmund, Norbert and Wąsowski, Andrzej},
booktitle={2013 28th IEEE/ACM International Conference on Automated Software Engineering (ASE)}, 
title={Variability-aware performance prediction: A statistical learning approach}, 
year={2013},
volume={},
number={},
pages={301-311},
doi={10.1109/ASE.2013.6693089}
}

Q-A : 
Yes, 6 software systems, including x264, llvm, Apache, SQLite

Q-B : 
No, same data as SPLConqueror

Q-C : 
No, the only mention to input is "All systems have been deployed and used in
real-world scenarios. Moreover, the performance is measured
by standard benchmarks."
It does not recognize the input sensitivity threat.

Q-D : 
No


### Paper 36

Title : 
An extensible framework for software configuration optimization on heterogeneous computing systems: Time and energy case study

Bibtex :
@article{SVOGOR201930,
title = {An extensible framework for software configuration optimization on heterogeneous computing systems: Time and energy case study},
journal = {Information and Software Technology},
volume = {105},
pages = {30-42},
year = {2019},
issn = {0950-5849},
doi = {https://doi.org/10.1016/j.infsof.2018.08.003},
url = {https://www.sciencedirect.com/science/article/pii/S0950584918301666},
author = {Ivan Švogor and Ivica Crnković and Neven Vrček},
keywords = {Cyber–physical systems, Software components, Power consumption, Execution time, Robot experiment, Heterogeneous computing, Component based software},
abstract = {Context: Application of component based software engineering methods to heterogeneous computing (HC) enables different software configurations to realize the same function with different non–functional properties (NFP). Finding the best software configuration with respect to multiple NFPs is a non–trivial task. Objective: We propose a Software Component Allocation Framework (SCAF) with the goal to acquire a (sub–) optimal software configuration with respect to multiple NFPs, thus providing performance prediction of a software configuration in its early design phase. We focus on the software configuration optimization for the average energy consumption and average execution time. Method: We validated SCAF through its instantiation on a real–world demonstrator and a simulation. Firstly, we verified the correctness of our model through comparing the performance prediction of six software configurations to the actual performance, obtained through extensive measurements with a confidence interval of 95%. Secondly, to demonstrate how SCAF scales up, we performed software configuration optimization on 55 generated use–cases (with solution spaces ranging from 1030 to 3070) and benchmark the results against best performing random configurations. Results: The performance of a configuration as predicted by our framework matched the configuration implemented and measured on a real–world platform. Furthermore, by applying the genetic algorithm and simulated annealing to the weight function given in SCAF, we obtain sub–optimal software configurations differing in performance at most 7% and 13% from the optimal configuration (respectfully). Conclusion: SCAF is capable of correctly describing a HC platform and reliably predict the performance of software configuration in the early design phase. Automated in the form of an Eclipse plugin, SCAF allows software architects to model architectural constraints and preferences, acting as a multi–criterion software architecture decision support system. In addition to said, we also point out several interesting research directions, to further investigate and improve our approach.}
}

Q-A : 
Yes, the robot here processes input images

Q-B : 
Yes, different images were recorded. The size of the input image fed to the algorithm processing it vary.

Q-C : 
No

Q-D : 
No 


### Paper 37

Title : 
Performance prediction using support vector machine for the configuration of optimization algorithms

Bibtex :
@INPROCEEDINGS{8284699,
author={Afia, Abdellatif El and Sarhani, Malek},
booktitle={2017 3rd International Conference of Cloud Computing Technologies and Applications (CloudTech)}, 
title={Performance prediction using support vector machine for the configuration of optimization algorithms}, 
year={2017},
volume={},
number={},
pages={1-7},
doi={10.1109/CloudTech.2017.8284699}
}

Q-A : 
Yes, it includes data measured on SAT solvers (i.e. that processes input data)

Q-B : 
"First of all, the paper takes the dataset evaluated by the
SatZilla solver These dataset have been used in the SAT
competition."
-> Yes, this dataset consider several SAT formulae

Q-C : 
No, no mention of input sensitivity

Q-D : 
"That is,
the aim of this dataset is to predict each algorithm (solver) per-
formance on the different instances. Therefore, the algorithm
ID is considered as a feature to determine the corresponding
algorithm for each instance of the problem." -> No


### Paper 38

Title : 
Autotuning Algorithmic Choice for Input Sensitivity

Bibtex :
@inproceedings{10.1145/2737924.2737969,
author = {Ding, Yufei and Ansel, Jason and Veeramachaneni, Kalyan and Shen, Xipeng and O’Reilly, Una-May and Amarasinghe, Saman},
title = {Autotuning Algorithmic Choice for Input Sensitivity},
year = {2015},
isbn = {9781450334686},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/2737924.2737969},
doi = {10.1145/2737924.2737969},
abstract = { A daunting challenge faced by program performance autotuning is input sensitivity,
where the best autotuned configuration may vary with different input sets. This paper
presents a novel two-level input learning algorithm to tackle the challenge for an
important class of autotuning problems, algorithmic autotuning. The new approach uses
a two-level input clustering method to automatically refine input grouping, feature
selection, and classifier construction. Its design solves a series of open issues
that are particularly essential to algorithmic autotuning, including the enormous
optimization space, complex influence by deep input features, high cost in feature
extraction, and variable accuracy of algorithmic choices. Experimental results show
that the new solution yields up to a 3x speedup over using a single configuration
for all inputs, and a 34x speedup over a traditional one-level method for addressing
input sensitivity in program optimizations. },
booktitle = {Proceedings of the 36th ACM SIGPLAN Conference on Programming Language Design and Implementation},
pages = {379–390},
numpages = {12},
keywords = {Algorithmic Selection, Input Sensitivity, Autotuning, Variable Accuracy},
location = {Portland, OR, USA},
series = {PLDI '15}
}

Q-A : 
Yes, the paper considers Petabricks, a compiler

Q-B : 
Yes, they consider multiple instances of inputs, justification here "To measure the efficacy of our system, we tested it on a suite of
6 parallel PetaBricks benchmarks [5]."

Q-C : 
Yes, the title is "Autotuning Algorithmic Choice for Input Sensitivity"

Q-D : 
Yes, the paper proposes a two-step approach that should be reproduced on other software systems. A promising approach to tune a software system for its input.


### Paper 39

Title : 
Learning Non-Deterministic Impact Models for Adaptation

Bibtex :
@inproceedings{10.1145/3194133.3194138,
author = {Duarte, Francisco and Gil, Richard and Romano, Paolo and Lopes, Ant\'{o}nia and Rodrigues, Lu\'{\i}s},
title = {Learning Non-Deterministic Impact Models for Adaptation},
year = {2018},
isbn = {9781450357159},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3194133.3194138},
doi = {10.1145/3194133.3194138},
abstract = {Many adaptive systems react to variations in their environment by changing their configuration.
Often, they make the adaptation decisions based on some knowledge about how the reconfiguration
actions impact the key performance indicators. However, the outcome of these actions
is typically affected by uncertainty. Adaptation actions have non-deterministic impacts,
potentially leading to multiple outcomes. When this uncertainty is not captured explicitly
in the models that guide adaptation, decisions may turn out ineffective or even harmful
to the system. Also critical is the need for these models to be interpretable to the
human operators that are accountable for the system. However, accurate impact models
for actions that result in non-deterministic outcomes are very difficult to obtain
and existing techniques that support the automatic generation of these models, mainly
based on machine learning, are limited in the way they learn non-determinism.In this
paper, we propose a method to learn human-readable models that capture non-deterministic
impacts explicitly. Additionally, we discuss how to exploit expert's knowledge to
bootstrap the adaptation process as well as how to use the learned impacts to revise
models defined offline. We motivate our work on the adaptation of applications in
the cloud, typically affected by hardware heterogeneity and resource contention. To
validate our approach we use a prototype based on the RUBiS auction application.},
booktitle = {Proceedings of the 13th International Conference on Software Engineering for Adaptive and Self-Managing Systems},
pages = {196–205},
numpages = {10},
keywords = {adaptive systems, runtime models, machine learning, uncertainty},
location = {Gothenburg, Sweden},
series = {SEAMS '18}
}

Q-A : 
Yes, RUBIs (apparently an e-commerce website prototype, mimicking ebay according to https://github.com/cloud-control/brownout-rubis , the link given in the paper is dead for me). So it does have input data (e.g. the content of the webpage)

Q-B : 
Yes,
"We used Autobench5 3 as
a benchmark workload generator and drive httperf [25] to issue
the requests"

Q-C : 
Yes,
"Moreover, deci-
sions can derive into separate policies that depend on the workload
increase ranges" 
The way they define resource contention may look like it is just an IO problem, but looks like the input sensitivity?

Q-D : 
Yes, it is applied to inputs. They cut the input space into different area of range? Looks like a classification of inputs before the prediction.
Then, a probability is given for an action (classic for self-adaptative systems).

### Paper 40

Title : 
Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification Algorithms

Bibtex :
@inproceedings{10.1145/2487575.2487629,
author = {Thornton, Chris and Hutter, Frank and Hoos, Holger H. and Leyton-Brown, Kevin},
title = {Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification Algorithms},
year = {2013},
isbn = {9781450321747},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/2487575.2487629},
doi = {10.1145/2487575.2487629},
abstract = {Many different machine learning algorithms exist; taking into account each algorithm's
hyperparameters, there is a staggeringly large number of possible alternatives overall.
We consider the problem of simultaneously selecting a learning algorithm and setting
its hyperparameters, going beyond previous work that attacks these issues separately.
We show that this problem can be addressed by a fully automated approach, leveraging
recent innovations in Bayesian optimization. Specifically, we consider a wide range
of feature selection techniques (combining 3 search and 8 evaluator methods) and all
classification approaches implemented in WEKA's standard distribution, spanning 2
ensemble methods, 10 meta-methods, 27 base classifiers, and hyperparameter settings
for each classifier. On each of 21 popular datasets from the UCI repository, the KDD
Cup 09, variants of the MNIST dataset and CIFAR-10, we show classification performance
often much better than using standard selection and hyperparameter optimization methods.
We hope that our approach will help non-expert users to more effectively identify
machine learning algorithms and hyperparameter settings appropriate to their applications,
and hence to achieve improved performance.},
booktitle = {Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
pages = {847–855},
numpages = {9},
keywords = {weka, hyperparameter optimization, model selection},
location = {Chicago, Illinois, USA},
series = {KDD '13}
}

Q-A : 
Yes, we can consider that Auto-Weka takes a dataset as "input", and tells which algorithm is best adapted to the case. In that sense, it has some inputs.

Q-B : 
"We evaluated Auto-WEKA on 21 prominent benchmark
datasets (see Table 3): 15 sets from the UCI repository [11];
the ‘convex’, ‘MNIST basic’ and ‘rotated MNIST with back-
ground images’ tasks used in [4]; the appentency task from
the KDD Cup ’09; and two versions of the CIFAR-10 im-
age classification task [19] (CIFAR-10-Small is a subset of
CIFAR-10, where only the first 10 000 training data points
are used rather than the full 50 000.)"
-> Yes

Q-C : 
Yes, definitely
"Furthermore, there was
no single method that achieved good performance across
all datasets: every method was at least 22% worse than
the best for at least one data set. We conclude that some
form of algorithm selection is essential for achieving good
performance"
It is a quite common comment in many papers (e.g. paper 25 does the same remark). They try to quantify how it varies with inputs, which is quite the same as our RQ-C.

Q-D : 
No, it is dataset per dataset (but the algorithm selection part is interesting and looks like a first step of classification).
Very interesting to test in terms of input sensitivity.


### Paper 41

Title : 
Predicting performance via automated feature-interaction detection

Bibtex :
@inproceedings{siegmund2012predicting,
  title={Predicting performance via automated feature-interaction detection},
  author={Siegmund, Norbert and Kolesnikov, Sergiy S and K{\"a}stner, Christian and Apel, Sven and Batory, Don and Rosenm{\"u}ller, Marko and Saake, Gunter},
  booktitle={2012 34th International Conference on Software Engineering (ICSE)},
  pages={167--177},
  year={2012},
  organization={IEEE}
}

Q-A : 
Yes, different configurable systems like SQLite, LLVM, x264 

Q-B : 
No for x264, only one input
No for llvm (the whole test suite and no detail)
Yes for Apache (use of autobench)
-> Yes

Q-C : 
"Besides the target platform, other factors influence non-
functional properties of a program. Database performance
depends on the workload, cache size, page size, disk speed,
reliability and security features, and so forth."
-> Yes

also interesting:
"We did not develop our own benchmark to avoid
bias and uncommon performance behavior caused by flaws
in benchmark designs."

Q-D : 
No


### Paper 42

Title : 
SPL Conqueror: Toward Optimization of Non-Functional Properties in Software Product Lines

Bibtex :
@article{10.1007/s11219-011-9152-9,
author = {Siegmund, Norbert and Rosenm\"{u}ller, Marko and Kuhlemann, Martin and K\"{a}stner, Christian and Apel, Sven and Saake, Gunter},
title = {SPL Conqueror: Toward Optimization of Non-Functional Properties in Software Product Lines},
year = {2012},
issue_date = {September 2012},
publisher = {Kluwer Academic Publishers},
address = {USA},
volume = {20},
number = {3–4},
issn = {0963-9314},
url = {https://doi.org/10.1007/s11219-011-9152-9},
doi = {10.1007/s11219-011-9152-9},
abstract = {A software product line (SPL) is a family of related programs of a domain. The programs
of an SPL are distinguished in terms of features, which are end-user visible characteristics
of programs. Based on a selection of features, stakeholders can derive tailor-made
programs that satisfy functional requirements. Besides functional requirements, different
application scenarios raise the need for optimizing non-functional properties of a
variant. The diversity of application scenarios leads to heterogeneous optimization
goals with respect to non-functional properties (e.g., performance vs. footprint vs.
energy optimized variants). Hence, an SPL has to satisfy different and sometimes contradicting
requirements regarding non-functional properties. Usually, the actually required non-functional
properties are not known before product derivation and can vary for each application
scenario and customer. Allowing stakeholders to derive optimized variants requires
us to measure non-functional properties after the SPL is developed. Unfortunately,
the high variability provided by SPLs complicates measurement and optimization of
non-functional properties due to a large variant space. With SPL Conqueror, we provide
a holistic approach to optimize non-functional properties in SPL engineering. We show
how non-functional properties can be qualitatively specified and quantitatively measured
in the context of SPLs. Furthermore, we discuss the variant-derivation process in
SPL Conqueror that reduces the effort of computing an optimal variant. We demonstrate
the applicability of our approach by means of nine case studies of a broad range of
application domains (e.g., database management and operating systems). Moreover, we
show that SPL Conqueror is implementation and language independent by using SPLs that
are implemented with different mechanisms, such as conditional compilation and feature-oriented
programming.},
journal = {Software Quality Journal},
month = sep,
pages = {487–517},
numpages = {31},
keywords = {Feature-oriented software development, Measurement and optimization, SPL Conqueror, Non-functional properties, Software product lines}
}

Q-A : 
Yes, including SQLite and Berkeley DB

Q-B : 
This citation "For instance, we measure the time for sorting of the LinkedList SPL and we use Oracle’s standard read benchmark for Berkeley DB"
implies a Yes?

Q-C : 
No threat section, limitations are embedded in the results and in discussion. I did not found any reference to input sensitivity, which can be explained by the fact that the paper focus remains the considered variants, and not inputs at all.
No

Q-D : 
No

### Paper 43

Title : 
Automated Inference of Goal-Oriented Performance Prediction Functions

Bibtex :
@inproceedings{10.1145/2351676.2351703,
author = {Westermann, Dennis and Happe, Jens and Krebs, Rouven and Farahbod, Roozbeh},
title = {Automated Inference of Goal-Oriented Performance Prediction Functions},
year = {2012},
isbn = {9781450312042},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/2351676.2351703},
doi = {10.1145/2351676.2351703},
abstract = { Understanding the dependency between performance metrics (such as response time)
and software configuration or usage parameters is crucial in improving software quality.
However, the size of most modern systems makes it nearly impossible to provide a complete
performance model. Hence, we focus on scenario-specific problems where software engineers
require practical and efficient approaches to draw conclusions, and we propose an
automated, measurement-based model inference method to derive goal-oriented performance
prediction functions. For the practicability of the approach it is essential to derive
functional dependencies with the least possible amount of data. In this paper, we
present different strategies for automated improvement of the prediction model through
an adaptive selection of new measurement points based on the accuracy of the prediction
model. In order to derive the prediction models, we apply and compare different statistical
methods. Finally, we evaluate the different combinations based on case studies using
SAP and SPEC benchmarks. },
booktitle = {Proceedings of the 27th IEEE/ACM International Conference on Automated Software Engineering},
pages = {190–199},
numpages = {10},
keywords = {Model Inference, Performance Prediction},
location = {Essen, Germany},
series = {ASE 2012}
}

Q-A : 
Yes. Justification:
"In this case study, we address the problem of customizing
an SAP ERP application to an expected customer work-
load."

Q-B : 
"To generate load on the system we used the SAP
Sales and Distribution (SD) Benchmark."
Yes, they vary the load.

Q-C : 
In a way, Q-A's citation should be enough to say Yes, but not mention to the difference of performance between worklaods.
No?

Q-D :
"Instead, the goal is to provide a prac-
tical automated evaluation that helps the administrator to
determine the optimal allocation of work process for a given
workload type and a given system configuration."
-> No

### Paper 44

Title : 
White-Box Analysis over Machine Learning: Modeling Performance of Configurable Systems

Bibtex :
@misc{velez2021whitebox,
      title={White-Box Analysis over Machine Learning: Modeling Performance of Configurable Systems}, 
      author={Miguel Velez and Pooyan Jamshidi and Norbert Siegmund and Sven Apel and Christian Kästner},
      year={2021},
      eprint={2101.05362},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}

Q-A : 
Yes, including a Berkeley DB (input=database) and Density Converter (input=image)

Q-B : 
Different workloads for Berkeley
"To reduce cost, we separate the iter-
ative analysis and performance measurement in two steps and
perform the former with a drastically reduced workload size."
+
"For each system, we changed the workload parameter to run
the iterative dynamic taint analysis with a smaller workload
by factors ranging from 20 to 50000, depending on the
system."

And the appendix pdf here https://raw.githubusercontent.com/miguelvelezmj25/icse21-sm/master/appendix.pdf
details all the benchmarks, e.g.
"a publicly available 5616 ×3744 pixel JPEG photo." for Density Converter
"we executed the RunBenchC benchmark, shipped with the system, which is similar to the TPC-C benchmark." for H2
etc.
Definitely yes, it could be part of our protocol if they train a performance model each time.

Q-C : 
No, not really the focus of the paper, neither in threats nor in dicussions.
Closest citation to input sensitivity
"[...], the time to run the taint analysis with
the regular workload is extremely expensive; 11 hours instead
of 29 minutes to run the same 26 configurations. In fact, the
iterative analysis did no finish executing after 24 hours in the
other subject systems!"

Q-D : 
No ?

### Paper 45

Title : 
Sampling Effect on Performance Prediction of Configurable Systems: A Case Study

Bibtex :
@inproceedings{10.1145/3358960.3379137,
author = {Alves Pereira, Juliana and Acher, Mathieu and Martin, Hugo and J\'{e}z\'{e}quel, Jean-Marc},
title = {Sampling Effect on Performance Prediction of Configurable Systems: A Case Study},
year = {2020},
isbn = {9781450369916},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3358960.3379137},
doi = {10.1145/3358960.3379137},
abstract = {Numerous software systems are highly configurable and provide a myriad of configuration
options that users can tune to fit their functional and performance requirements (e.g.,
execution time). Measuring all configurations of a system is the most obvious way
to understand the effect of options and their interactions, but is too costly or infeasible
in practice. Numerous works thus propose to measure only a few configurations (a sample)
to learn and predict the performance of any combination of options' values. A challenging
issue is to sample a small and representative set of configurations that leads to
a good accuracy of performance prediction models. A recent study devised a new algorithm,
called distance-based sampling, that obtains state-of-the-art accurate performance
predictions on different subject systems. In this paper, we replicate this study through
an in-depth analysis of x264, a popular and configurable video encoder. We systematically
measure all 1,152 configurations of x264 with 17 input videos and two quantitative
properties (encoding time and encoding size). Our goal is to understand whether there
is a dominant sampling strategy over the very same subject system (x264), i.e., whatever
the workload and targeted performance properties. The findings from this study show
that random sampling leads to more accurate performance models. However, without considering
random, there is no single "dominant" sampling, instead different strategies perform
best on different inputs and non-functional properties, further challenging practitioners
and researchers.},
booktitle = {Proceedings of the ACM/SPEC International Conference on Performance Engineering},
pages = {277–288},
numpages = {12},
keywords = {software product lines, configurable systems, machine learning, performance prediction},
location = {Edmonton AB, Canada},
series = {ICPE '20}
}

Q-A : 
Yes, including x264

Q-B : 
Yes,  17 input videos for x264, see Table 1

Q-C : 
Yes. Justification:
It is a reproduction of a study on sampling effect including different input videos and showing that results can depend on input videos. 
In addition,
"While the seventeen videos cover a wide range of different input particularities and provided consistent
results across the experiments, this is a preliminary study in this direction and future work should consider additional
inputs based on an in-depth qualitative study of video characteristics."

Q-D : 
No, not the point of the paper

### Paper 46

Title : 
Perf-AL: Performance Prediction for Configurable Software through Adversarial Learning

Bibtex :
@inproceedings{10.1145/3382494.3410677,
author = {Shu, Yangyang and Sui, Yulei and Zhang, Hongyu and Xu, Guandong},
title = {Perf-AL: Performance Prediction for Configurable Software through Adversarial Learning},
year = {2020},
isbn = {9781450375801},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3382494.3410677},
doi = {10.1145/3382494.3410677},
abstract = {Context: Many software systems are highly configurable. Different configuration options
could lead to varying performances of the system. It is difficult to measure system
performance in the presence of an exponential number of possible combinations of these
options.Goal: Predicting software performance by using a small configuration sample.Method:
This paper proposes Perf-AL to address this problem via adversarial learning. Specifically,
we use a generative network combined with several different regularization techniques
(L1 regularization, L2 regularization and a dropout technique) to output predicted
values as close to the ground truth labels as possible. With the use of adversarial
learning, our network identifies and distinguishes the predicted values of the generator
network from the ground truth value distribution. The generator and the discriminator
compete with each other by refining the prediction model iteratively until its predicted
values converge towards the ground truth distribution.Results: We argue that (i) the
proposed method can achieve the same level of prediction accuracy, but with a smaller
number of training samples. (ii) Our proposed model using seven real-world datasets
show that our approach outperforms the state-of-the-art methods. This help to further
promote software configurable performance.Conclusion: Experimental results on seven
public real-world datasets demonstrate that PERF-AL outperforms state-of-the-art software
performance prediction methods.},
booktitle = {Proceedings of the 14th ACM / IEEE International Symposium on Empirical Software Engineering and Measurement (ESEM)},
articleno = {16},
numpages = {11},
keywords = {Software performance prediction, adversarial learning, configurable systems, regularization},
location = {Bari, Italy},
series = {ESEM '20}
}

Q-A : 
Yes, LLVM and Apache at least

Q-B : 
No, not mentioned.
They use data from ref 27, which is our 17th paper. See the justification for the paper 17.

Q-C : 
No, not throughout the paper and not in 4.8

Q-D : 
No

### Paper 47

Title : 
Mastering Uncertainty in Performance Estimations of Configurable Software Systems

Bibtex :
@inproceedings{10.1145/3324884.3416620,
author = {Dorn, Johannes and Apel, Sven and Siegmund, Norbert},
title = {Mastering Uncertainty in Performance Estimations of Configurable Software Systems},
year = {2020},
isbn = {9781450367684},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3324884.3416620},
doi = {10.1145/3324884.3416620},
abstract = {Understanding the influence of configuration options on performance is key for finding
optimal system configurations, system understanding, and performance debugging. In
prior research, a number of performance-influence modeling approaches have been proposed,
which model a configuration option's influence and a configuration's performance as
a scalar value. However, these point estimates falsely imply a certainty regarding
an option's influence that neglects several sources of uncertainty within the assessment
process, such as (1) measurement bias, (2) model representation and learning process,
and (3) incomplete data. This leads to the situation that different approaches and
even different learning runs assign different scalar performance values to options
and interactions among them. The true influence is uncertain, though. There is no
way to quantify this uncertainty with state-of-the-art performance modeling approaches.
We propose a novel approach, P4, based on probabilistic programming that explicitly
models uncertainty for option influences and consequently provides a confidence interval
for each prediction of a configuration's performance alongside a scalar. This way,
we can explain, for the first time, why predictions may cause errors and which option's
influences may be unreliable. An evaluation on 12 real-world subject systems shows
that P4's accuracy is in line with the state of the art while providing reliable confidence
intervals, in addition to scalar predictions.},
booktitle = {Proceedings of the 35th IEEE/ACM International Conference on Automated Software Engineering},
pages = {684–696},
numpages = {13},
keywords = {probabilistic programming, configurable software systems, performance-influence modeling, P4},
location = {Virtual Event, Australia},
series = {ASE '20}
}

Q-A : 
Yes, including 7z, x264, vp9, etc.

Q-B : 
Seems to be only one workload. Possible justification here :
"We measure the performance of a configuration by configuring a software system, and executing a workload."

But there might have other workloads. When looking at https://archive.softwareheritage.org/browse/directory/5a525f45ec77dbe982081e7f8159e9541391725e/?path=data
it is not obvious to clarify this point

No

Q-C : 
No, no mention in threats, and none read throughout the paper.

Q-D : 
No, for this paper Q-B=No implies Q-D=No


### Paper 48

Title : 
The Interplay of Sampling and Machine Learning for Software Performance Prediction

Bibtex :
@ARTICLE{9062326,
author={Kaltenecker, Christian and Grebhahn, Alexander and Siegmund, Norbert and Apel, Sven},
journal={IEEE Software}, 
title={The Interplay of Sampling and Machine Learning for Software Performance Prediction}, 
year={2020},
volume={37},
number={4},
pages={58-66},
doi={10.1109/MS.2020.2987024}
}

Q-A : 
Yes, Dune (input = images) and VP9 (input = videos)

Q-B : 
No, not really the purpose of the paper; hard to combine sampling, learning and inputs.

Q-C : 
No, none I noticed

Q-D : 
No, implied by Q-B=No

### Paper 49

Title :
Whence to Learn? Transferring Knowledge in Configurable Systems using BEETLE

Bibtex :
@ARTICLE{9050841,
author={Krishna, Rahul and Nair, Vivek and Jamshidi, Pooyan and Menzies, Tim},
journal={IEEE Transactions on Software Engineering}, 
title={Whence to Learn? Transferring Knowledge in Configurable Systems using BEETLE}, 
year={2020},
volume={},
number={},
pages={1-1},
doi={10.1109/TSE.2020.2983927}
}

Q-A : 
Yes, justification:
"For evaluation, we explore five real-world software sys-
tems from different domains– a video encoder, a SAT solver,
a SQL database, a high-performance C-compiler, and a
streaming data analytics tool (measured under 57 enviro-
ments overall)."

Q-B : 
Yes, see Table 1 combined with
"The
performance of each of the |C|configurations are measured
under different hardware (H), workloads (W), and software
versions (V ). A unique combination of H,W,V constitutes
an enviroment which is denoted by E"
The different environments include different workloads 

Q-C : 
Yes, justification:
"There is a possibility that measurement
of other performance measures or availability of additional
performance measures may result in a different outcome.
Therefore, one has to be careful when generalizing our
findings to other subject systems and environment changes"

Q-D : 
Indeed a trasnfer-learning technique based on source selection to test on the different input goups or profiles shown in RQ-A, that can solve the input sensivitity problem.
Yes

### Paper 50

Title : 
White-Box Performance-Influence Models: A Profiling and Learning Approach

Bibtex : 
Only the replication package in IEEE for now, to update later
@misc{weber2021whitebox,
title={White-Box Performance-Influence Models: A Profiling and Learning Approach}, 
author={Max Weber and Sven Apel and Norbert Siegmund},
year={2021},
eprint={2102.06395},
archivePrefix={arXiv},
primaryClass={cs.SE}
}

Q-A : 
Yes, H2 (input=database) and DC (input = images)

Q-B : 
Yes, justification:
"The BATIK rasterizer converts SVG files to a raster format.
As workload, we used the DACAPO benchmark suite [55],
which contains a set of SVG images of different sizes that
can be used for performance tests"
and confirmed here 
https://archive.softwareheritage.org/browse/revision/2e61f8ce57498194c2af0cd76e87498a174f07fa/?path=supplementary-website/experiment_data/data

Q-C : 
No, not in threats and not throughout this paper

Q-D : 
No, not the point of the paper


### Paper 51

Title : 
Identifying Software Performance Changes Across Variants and Versions

Bibtex :
@INPROCEEDINGS{9285664, 
author={Mühlbauer, Stefan and Apel, Sven and Siegmund, Norbert},  booktitle={2020 35th IEEE/ACM International Conference on Automated Software Engineering (ASE)},
title={Identifying Software Performance Changes Across Variants and Versions},
year={2020},
volume={},
number={},
pages={611-622},
doi={}
}

Q-A : 
Yes, including xz and lrzip (input = file system), but also oggenc (input = audio files), which is quite novel

Q-B : 
Yes, 
e.g. for lrzip
"As a workload for the file compression tools lrzip and xz, we used
the Silesia corpus, which contains over 200 MB of files of different
types."
and for oggenc
"For the
audio transcoder oggenc, we encoded a raw WAVE audio file of
over 60 MB from the Wikimedia Commons collection."

Q-C : 
"we were able to test corner
cases and assess scalability so that we believe that our results hold
for many practical use cases."
is close, but no?

Q-D : 
No, not really the point of the paper, more about the evolution of software in general, this could be combined with input sensitivity -> i.e. how does the input sensitivity evolve with versions and variants?


### Paper 52

Title : 
Automated Performance Tuning for Highly-Configurable Software Systems

Bibtex :
Only arxiv for now 
@article{han2020automated,
  title={Automated Performance Tuning for Highly-Configurable Software Systems},
  author={Han, Xue and Yu, Tingting},
  journal={arXiv preprint arXiv:2010.01397},
  year={2020}
}

Q-A : 
Yes, including Apache and MySQL

Q-B : 
Yes, justification:
"Benchmark tools are used to
generate workload and report performance measurements. For
instance, the Apache Benchmark (ab) is used to measure
Apache’s performance by the following command: “ab -n 1000
-c 10 http:localhost”. -n specifies the number of requests and
-c specifies the level of concurrency. The benchmark tools
provide an elegant solution to generate synthetic traffic on
demand."

Q-C : 
No mention of input sensitivity, no?

Q-D : 
No

### Paper 53

Title : 
ConfProf: White-Box Performance Profiling of Configuration Options

Bibtex :
@inproceedings{10.1145/3427921.3450255,
author = {Han, Xue and Yu, Tingting and Pradel, Michael},
title = {ConfProf: White-Box Performance Profiling of Configuration Options},
year = {2021},
isbn = {9781450381949},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3427921.3450255},
doi = {10.1145/3427921.3450255},
abstract = {Modern software systems are highly customizable through configuration options. The
sheer size of the configuration space makes it challenging to understand the performance
influence of individual configuration options and their interactions under a specific
usage scenario. Software with poor performance may lead to low system throughput and
long response time. This paper presents ConfProf, a white-box performance profiling
technique with a focus on configuration options. ConfProf helps developers understand
how configuration options and their interactions influence the performance of a software
system. The approach combines dynamic program analysis, machine learning, and feedback-directed
configuration sampling to profile the program execution and analyze the performance
influence of configuration options. Compared to existing approaches, ConfProf uses
a white-box approach combined with machine learning to rank performance-influencing
configuration options from execution traces. We evaluate the approach with 13 scenarios
of four real-world, highly-configurable software systems. The results show that ConfProf
ranks performance-influencing configuration options with high accuracy and outperform
a state of the art technique.},
booktitle = {Proceedings of the ACM/SPEC International Conference on Performance Engineering},
pages = {1–8},
numpages = {8},
keywords = {software performance, performance profiling},
location = {Virtual Event, France},
series = {ICPE '21}
}

Q-A : 
Yes, including Apache Server, PBZIP2 and PostgreSQL

Q-B : 
They vary the scenarii i.e. compress or decompress data,
e.g.
"For instance, PBZIP2 uses the configuration option -z
to compress data, whereas the configuration option -d is used for
decompressing data. Since performance bugs often require specific
workloads to manifest, we identify the workload for each usage
scenario accordingly."

but no different inputs
e.g.
"The input to ConfProf is a configurable program
and a usage scenario that exercises the program"

Q-C : 
Yes,
"Similar to existing profiling techniques,
ConfProf is based on dynamic analysis and therefore limited to
observing the executions triggered by a given set of inputs. The
problem of finding suitable inputs for performance analysis [6, 8, 35]
is orthogonal to the issue addressed here."

Q-D : 
No, see Q-C's answer

### Paper 54

Title : 
Transferring Pareto Frontiers across Heterogeneous Hardware Environments

Bibtex :
@inproceedings{10.1145/3358960.3379127,
author = {Valov, Pavel and Guo, Jianmei and Czarnecki, Krzysztof},
title = {Transferring Pareto Frontiers across Heterogeneous Hardware Environments},
year = {2020},
isbn = {9781450369916},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3358960.3379127},
doi = {10.1145/3358960.3379127},
abstract = {Software systems provide user-relevant configuration options called features. Features
affect functional and non-functional system properties, whereas selections of features
represent system configurations. A subset of configuration space forms a Pareto frontier
of optimal configurations in terms of multiple properties, from which a user can choose
the best configuration for a particular scenario. However, when a well-studied system
is redeployed on a different hardware, information about property value and the Pareto
frontier might not apply. We investigate whether it is possible to transfer this information
across heterogeneous hardware environments. We propose a methodology for approximating
and transferring Pareto frontiers of configurable systems across different hardware
environments. We approximate a Pareto frontier by training an individual predictor
model for each system property, and by aggregating predictions of each property into
an approximated frontier. We transfer the approximated frontier across hardware by
training a transfer model for each property, by applying it to a respective predictor,
and by combining transferred properties into a frontier. We evaluate our approach
by modeling Pareto frontiers as binary classifiers that separate all system configurations
into optimal and non-optimal ones. Thus we can assess quality of approximated and
transferred frontiers using common statistical measures like sensitivity and specificity.
We test our approach using five real-world software systems from the compression domain,
while paying special attention to their performance. Evaluation results demonstrate
that accuracy of approximated frontiers depends linearly on predictors' training sample
sizes, whereas transferring introduces only minor additional error to a frontier even
for small training sizes.},
booktitle = {Proceedings of the ACM/SPEC International Conference on Performance Engineering},
pages = {12–23},
numpages = {12},
keywords = {linear regression, regression trees, performance prediction, Pareto frontier transferring, configurable software, Pareto frontier},
location = {Edmonton AB, Canada},
series = {ICPE '20}
}

Q-A : 
Yes, including x264, bzip, etc.

Q-B : 
No, each system has its own workload, but fixed. Justification:
"We benchmarked BZIP2, GZIP, and XZ using large text compression benchmark [22],
which represents first 109 bytes of Wikipedia XML archive. We
benchmarked FLAC using Ghosts I-IV [23] music album of a band
‘Nine Inch Nails’, that contains 36 tracks of improvisation music,
released under Creative Commons license. We benchmarked x264
using a trailer of Sintel [1] open-content film created and released
under Creative Commons license by Blender Foundation."

Besides, https://bitbucket.org/valovp/icpe2020 is not directly available:
I created an account to see the repo and I got the following error message
"We can't let you see this page

To access this page, you may need to log in with another account. You can also return to the previous page or go back to your dashboard."

Q-C : 
No, not in threats to validity

Q-D : 
It could be used to transfer the Pareto Frontier from one input to another by replacing the hardware/environment with the input data.
-> Yes

### Paper 55

Title : 
Deffe: A Data-Efficient Framework for Performance Characterization in Domain-Specific Computing

Bibtex :
@inproceedings{10.1145/3387902.3392633,
author = {Liu, Frank and Miniskar, Narasinga Rao and Chakraborty, Dwaipayan and Vetter, Jeffrey S.},
title = {Deffe: A Data-Efficient Framework for Performance Characterization in Domain-Specific Computing},
year = {2020},
isbn = {9781450379564},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3387902.3392633},
doi = {10.1145/3387902.3392633},
abstract = {As the computer architecture community moves toward the end of traditional device
scaling, domain-specific architectures are becoming more pervasive. Given the number
of diverse workloads and emerging heterogeneous architectures, exploration of this
design space is a constrained optimization problem in a high-dimensional parameter
space. In this respect, predicting workload performance both accurately and efficiently
is a critical task for this exploration. In this paper, we present Deffe: a framework
to estimate workload performance across varying architectural configurations. Deffe
uses machine learning to improve the performance of this design space exploration.
By casting the work of performance prediction itself as transfer learning tasks, the
modelling component of Deffe can leverage the learned knowledge on one workload and
"transfer" it to a new workload. Our extensive experimental results on a contemporary
architecture toolchain (RISC-V and GEM5) and infrastructure show that the method can
achieve superior testing accuracy with an effective reduction of 32-80\texttimes{} in terms of
the amount of required training data. The overall run-time can be reduced from 400
hours to 5 hours when executed over 24 CPU cores. The infrastructure component of
Deffe is based on scalable and easy-to-use open-source software components.},
booktitle = {Proceedings of the 17th ACM International Conference on Computing Frontiers},
pages = {182–191},
numpages = {10},
keywords = {machine learning, transfer learning, RISC-V, multichannel convolution, workload characterization},
location = {Catania, Sicily, Italy},
series = {CF '20}
}

Q-A : 
"The workload prediction accuracy of the proposed machine learning
model was compared with three baseline machine learning regres-
sion methods"

Here the ML methods are like the configurable systems, and they process data
-> Yes

Q-B : 
Yes, justification:
"The benchmarks considered for the experiments are Back-
propagation (backprop), KMeans clustering (k-means), and Needleman-
Wunsch (nw) kernels, which are obtained from the Rodinia bench-
mark suite [10], along with the matrix multiplication (matmul)
kernel."

Q-C : 
See the keyword "workload characterization" in the paper at the beginning.
Definitely the underlying motivation of this paper.

Another justification:
"The selected benchmarks have run-time kernel parameters
that affect the run-time performance of the kernel"

-> Yes

Q-D : 
Yes, a transfer learning technique to apply in another domain.

### Paper 56

Title : 
On the Use of ML for Blackbox System Performance Prediction

Bibtex :
@inproceedings {265059,
author = {Silvery Fu and Saurabh Gupta and Radhika Mittal and Sylvia Ratnasamy},
title = {On the Use of {ML} for Blackbox System Performance Prediction},
booktitle = {18th {USENIX} Symposium on Networked Systems Design and Implementation ({NSDI} 21)},
year = {2021},
isbn = {978-1-939133-21-2},
pages = {763--784},
url = {https://www.usenix.org/conference/nsdi21/presentation/fu},
publisher = {{USENIX} Association},
month = apr,
}

Q-A : 
Yes, justification:
"(i) Application-level input parameters capture inputs that
the application acts on;"

Q-B : 
Yes, justification:
"We experiment with
varying the size of these inputs on a scale of 1 to 10, with
scale 1 being the default input size in the workload generator;"

Q-C : 
Yes. Justification:
"(i) Application-level input parameters capture inputs that
the application acts on; e.g., the records being sorted, or the
images being classified. We consider both the size of these
inputs and (when noted) the actual values of these inputs."


Q-D : 
"Our second assumption is that, for a given input dataset size,
the application’s input data is identical across all experiments.
I.e., repeated runs of a test configuration act on the same input
data. E.g., all training/test data for an application that sorts N
records, will involve exactly the same N records. We call this
our identical-inputs assumption."
The performance model that is trained will generalize to all inputs that respecting the identical-inputs assumption, but input sensitivity is not respecting the identical-inputs assumption, so no.

### Paper 57

Title : 
Source Selection in Transfer Learning for Improved Service Performance Predictions

Bibtex :
@INPROCEEDINGS{9472818,
author={Larsson, Hannes and Taghia, Jalil and Moradi, Farnaz and Johnsson, Andreas},
booktitle={2021 IFIP Networking Conference (IFIP Networking)}, 
title={Source Selection in Transfer Learning for Improved Service Performance Predictions}, 
year={2021},
volume={},
number={},
pages={1-9},
doi={10.23919/IFIPNetworking52078.2021.9472818}
}

Q-A : 
Yes, including a Video-on-Demand (VoD) service  (according to the authors "modified VLC media player software") and a Key-Value Store (KVS) service. 
Input data are operations sent by fake clients.

Q-B : 
Yes justification:
"The KVS load generator controls the rate of KVS operations
issued per second. Both generators produce load according to
two distinct load patterns described below"

Q-C : 
"Under the assumption that there are sufficient number of
samples in source domains, one can robustly measure diversity
of the source domains. In this work, we used the Shannon
differential entropy as the measure of diversity, which is
one approach amongst other approaches."
-> Yes

also an interesting paper to read in order to understand how input sensitivity affects the way we should select the right source for a given target.

Q-D : 
Yes, it is a transfer learning based on source selection eg like paper 49 introducing Beetle.

### Paper 58

Title : 
Efficient Compiler Autotuning via Bayesian Optimization

Bibtex :
@INPROCEEDINGS{9401979,
author={Chen, Junjie and Xu, Ningxin and Chen, Peiqi and Zhang, Hongyu},
booktitle={2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE)}, 
title={Efficient Compiler Autotuning via Bayesian Optimization}, 
year={2021},
volume={},
number={},
pages={1198-1209},
doi={10.1109/ICSE43902.2021.00110}
}

Q-A : 
Yes, gcc and llvm

Q-B : 
Yes, two C benchmarks (i.e., cBench and PolyBench)

Q-C : 
Yes,
"(e.g., only a small number of optimization flags, referred to as
impactful optimizations, can have noticeable impact on the
runtime performance of a specific program). As a result,
the direct application of the existing Bayesian optimization
methods in compiler autotuning is not efficient (which will be
demonstrated in our study presented in Section V-B)."

Q-D : 
"many compiler autotuning approaches have been proposed to automatically tune optimization flags in order to achieve required runtime performance for a given program"
If the method proposed in this paper outperforms other SOTA methods, it is yet applicable on a per-input basis.
We do not use prior knowledge (i.e. of other input programs) to compile the current one.
For this reason, we answer "no", even if this is a promising autotuning technique.

### Paper 59

Title : 
On Using Retrained and Incremental Machine Learning for Modeling Performance of Adaptable Software: An Empirical Comparison

Bibtex :
@INPROCEEDINGS{8787029,
  author={Chen, Tao},
  booktitle={2019 IEEE/ACM 14th International Symposium on Software Engineering for Adaptive and Self-Managing Systems (SEAMS)}, 
  title={All Versus One: An Empirical Comparison on Retrained and Incremental Machine Learning for Modeling Performance of Adaptable Software}, 
  year={2019},
  volume={},
  number={},
  pages={157-168},
  doi={10.1109/SEAMS.2019.00029}
}

Q-A : 
Yes, including RUBIs, Tomcat, mySQL

Q-B : 
Yes, justification:
"Notably, through different
workload patterns, traces and frequencies, we obtained a total
of 72 cases for each pair of performance indicator and learning
algorithm on S-RUBiS."
-> different cases of trained algorithm

Q-C : 
Not in threats, and none I read throughout the paper -> No

Q-D : 
No, one model for each case


### Paper 60

Title : 
DeepPerf: Performance Prediction for Configurable Software with Deep Sparse Neural Network

Bibtex :
@INPROCEEDINGS{8811988,
  author={Ha, Huong and Zhang, Hongyu},
  booktitle={2019 IEEE/ACM 41st International Conference on Software Engineering (ICSE)}, 
  title={DeepPerf: Performance Prediction for Configurable Software with Deep Sparse Neural Network}, 
  year={2019},
  volume={},
  number={},
  pages={1095-1106},
  doi={10.1109/ICSE.2019.00113}
}

Q-A : 
Yes, including x264, llvm, SQLite

Q-B : 
Note that input, in the paper, refers to configurations (logical for the authors, but might be confusing for us).

They mostly use the data from SPLConqueror, so no variation wtr input data.

Confirmed by looking at https://github.com/DeepPerf/DeepPerf/tree/master/Data
-> No

Q-C : 
Not in threats, not throughout the paper -> No

Q-D : 
No


### Paper 61

Title : 
DeepXplore: Automated Whitebox Testing of Deep Learning Systems

Bibtex :
@article{10.1145/3361566,
author = {Pei, Kexin and Cao, Yinzhi and Yang, Junfeng and Jana, Suman},
title = {DeepXplore: Automated Whitebox Testing of Deep Learning Systems},
year = {2019},
issue_date = {November 2019},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {62},
number = {11},
issn = {0001-0782},
url = {https://doi.org/10.1145/3361566},
doi = {10.1145/3361566},
abstract = {Deep learning (DL) systems are increasingly deployed in safety- and security-critical
domains such as self-driving cars and malware detection, where the correctness and
predictability of a system's behavior for corner case inputs are of great importance.
Existing DL testing depends heavily on manually labeled data and therefore often fails
to expose erroneous behaviors for rare inputs.We design, implement, and evaluate DeepXplore,
the first white-box framework for systematically testing real-world DL systems. First,
we introduce neuron coverage for measuring the parts of a DL system exercised by test
inputs. Next, we leverage multiple DL systems with similar functionality as cross-referencing
oracles to avoid manual checking. Finally, we demonstrate how finding inputs for DL
systems that both trigger many differential behaviors and achieve high neuron coverage
can be represented as a joint optimization problem and solved efficiently using gradient-based
search techniques.DeepXplore efficiently finds thousands of incorrect corner case
behaviors (e.g., self-driving cars crashing into guard rails and malware masquerading
as benign software) in state-of-the-art DL models with thousands of neurons trained
on five popular datasets such as ImageNet and Udacity self-driving challenge data.
For all tested DL models, on average, DeepXplore generated one test input demonstrating
incorrect behavior within one second while running only on a commodity laptop. We
further show that the test inputs generated by DeepXplore can also be used to retrain
the corresponding DL model to improve the model's accuracy by up to 3%.},
journal = {Commun. ACM},
month = oct,
pages = {137–145},
numpages = {9}
}

Q-A : 
A neural network, that takes a dataset as input, not conventional as system, but configurable and takes input data
-> Yes

Q-B : 
5 different datasets : MNIST, ImageNet, Driving, Contagio/VirusTotal,
and Drebin
-> Yes

Q-C : 
No, not mentioned

Q-D : 
It is not obvious to use it in this case to solve the input sensivity problem 
-> No

Check question : should we keep this paper in the list? (seems to be far from the problem we are facing)


### Paper 62

Title : 
Performance-Influence Model for Highly Configurable Software with Fourier Learning and Lasso Regression

Bibtex :
@INPROCEEDINGS{8919029,
author={Ha, Huong and Zhang, Hongyu},
booktitle={2019 IEEE International Conference on Software Maintenance and Evolution (ICSME)}, 
title={Performance-Influence Model for Highly Configurable Software with Fourier Learning and Lasso Regression}, 
year={2019},
volume={},
number={},
pages={470-480},
doi={10.1109/ICSME.2019.00080}
}

Q-A : 
Yes, including x264, llvm, SQLite

Q-B : 
"We use the ground-truth performance models described in [20]
and provided by the SPLConqueror project2."
Based on the previous explanations, no 

Q-C : 
No, not in threats, I did not read anything regarding input sensitivity in the paper

Q-D : 
No, Q-B=No implies Q-D=No for this paper


### Paper 63

Title : 
Transfer Learning for Cross-Model Regression in Performance Modeling for the Cloud

Bibtex :
@INPROCEEDINGS{8968941,
author={Iorio, Francesco and Hashemi, Ali B. and Tao, Michael and Amza, Cristiana},
booktitle={2019 IEEE International Conference on Cloud Computing Technology and Science (CloudCom)}, 
title={Transfer Learning for Cross-Model Regression in Performance Modeling for the Cloud}, 
year={2019},
volume={},
number={},
pages={9-18},
doi={10.1109/CloudCom.2019.00015}
}


Q-A : 
Yes, including MariaDB, a database manager system (input = database)

Q-B : 
Yes, justification:
"We measured and modeled the variation in the database’s
response throughput under varying configuration parameters,
both in terms of database configuration parameters and system-
level parameters. For data collection we configured the TPC-C
benchmark to populate and exercise a database representing,
alternatively, 10, 20, 40 and 64 warehouses, for a total database
size on disk of approximately 1.2GB to 6.4GB."
-> Yes, they vary the inputs in the experimental protocol

Q-C : 

Not directly in threats, but in future work, 
"Additionally, we plan to
study the relation between the dimensionality of the map and
its accuracy, and leverage sensitivity analysis of the system’s
performance function to select both the most appropriate map
dimensionality and its parameters"
This looks like a future work about sensitivity analysis i.e. database sensitivity, i.e. input sensitivity
-> Yes

Q-D : 
Yes, a transfer learning technique that could be used on our measurements.


### Paper 64

Title : 
SATUNE : A Study-Driven Auto-Tuning Approach for Configurable Software Verification Tools

Bibtex :
not published yet? -> ASE 2021

Q-A : 
Yes, the algorithm process programs (java programs, C programs)

Q-B : 
Yes, justification:
"we used all 368 benchmark programs
from SV-COMP 2019. These Java programs are written with
assertions, and the verification tools check if these asser-
tions always hold. Among the 368 programs, 204 (55.4%)
are known to be unsafe. For C tools, the SV-COMP 2018
benchmark has 9523 programs in total. 2 We randomly selected
a subset of 1000 programs that are subject to only one
verification check. Out of the 1000 programs we selected,
there are 335 programs that are subject to concurrency safety
verification, 41 to memory safety verification, 65 to integer
overflow verification, 485 to reachability verification, and 74
to verification of termination"
-> Yes, they vary the input programs in the experimental protocol

Q-C : 

Directly in the abstract:
"Examining the dataset, we find
there is generally no one-size-fits-all best configuration. Moreover,
a statistical analysis shows that many individual configuration
options do not have simple tradeoffs: they can be better or worse
depending on the program."
-> a big YES

Q-D : 
Yes, the purpose of the paper, SaTune!

### Paper 65

Title : 
Generalizable and Interpretable Learning for Configuration Extrapolation

Bibtex :
@inproceedings{10.1145/3468264.3468603,
author = {Ding, Yi and Pervaiz, Ahsan and Carbin, Michael and Hoffmann, Henry},
title = {Generalizable and Interpretable Learning for Configuration Extrapolation},
year = {2021},
isbn = {9781450385626},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3468264.3468603},
doi = {10.1145/3468264.3468603},
abstract = {Modern software applications are increasingly configurable, which puts a burden on users to tune these configurations for their target hardware and workloads. To help users, machine learning techniques can model the complex relationships between software configuration parameters and performance. While powerful, these learners have two major drawbacks: (1) they rarely incorporate prior knowledge and (2) they produce outputs that are not interpretable by users. These limitations make it difficult to (1) leverage information a user has already collected (e.g., tuning for new hardware using the best configurations from old hardware) and (2) gain insights into the learner’s behavior (e.g., understanding why the learner chose different configurations on different hardware or for different workloads). To address these issues, this paper presents two configuration optimization tools, GIL and GIL+, using the proposed generalizable and interpretable learning approaches. To incorporate prior knowledge, the proposed tools (1) start from known configurations, (2) iteratively construct a new linear model, (3) extrapolate better performance configurations from that model, and (4) repeat. Since the base learners are linear models, these tools are inherently interpretable. We enhance this property with a graphical representation of how they arrived at the highest performance configuration. We evaluate GIL and GIL+ by using them to configure Apache Spark workloads on different hardware platforms and find that, compared to prior work, GIL and GIL+ produce comparable, and sometimes even better performance configurations, but with interpretable results.},
booktitle = {Proceedings of the 29th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
pages = {728–740},
numpages = {13},
keywords = {generalizability, machine learning, Configuration, interpretability},
location = {Athens, Greece},
series = {ESEC/FSE 2021}
}

Q-A : 
Yes, Apache Spark

Q-B : 
We select ten Apache Spark workloads from the HiBench 4 big data
benchmark suite [ 14 ] -> Yes

Q-C : Yes, in the abstract
"Modern software applications are increasingly configurable, which
puts a burden on users to tune these configurations for their target
hardware and workloads."

Q-D : Yes


