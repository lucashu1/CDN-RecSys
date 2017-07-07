# CDN-RecSys: A Network That Works

An implementation of the LightFM hybrid recommender system for suggesting CDN (content delivery network) 
providers to various websites. 
Created at Tsinghua University's Big Data Technology R&D Center as part of USC Viterbi's Research Abroad program.

### Introduction

Spent 6 weeks in China!

Chinese government makes ICPs (internet content providers) register with info like their industry, 
so I figured I'd put that to use.

### Project Steps

1. Get up-to-date up on CDNs, Recommender Systems (literature review)
2. Obtain CDN/ICP dataset
3. Preprocess data: extract CDN-ICP interactions, user/item features
5. Create RecSys models
6. Evaluate RecSys models


### Results

Seemed to work pretty well. If I had more time, I would want to do some extra feature engineering.
Would also want to collect a lot more data from websites outside of China.


### Files Included

* LightFM.ipynb
* User/Item - Index Mappings (pkl)
* User/Item Feature Vectors (pkl)
* Train-test, Warm-cold interaction splits (pkl)


### Built With

* [LightFM](https://lyst.github.io/lightfm/docs/home.html): 
recommender system library for implicit feedback, hybrid models
* Pandas: for data pre-processing
* Scikit-optimize: for hyperparameter optimization
* Numpy/scipy: for pretty much everything


### Acknowledgments

* Maciej Kula (Lyst): LightFM creator
* Ethan Rosenthal: for his [blog posts](http://blog.ethanrosenthal.com) on recommender systems
* Zha Cong: my host/partner student at Tsinghua
* Professor Yin Hao, Professor Yu Longqiang (Tsinghua University)
