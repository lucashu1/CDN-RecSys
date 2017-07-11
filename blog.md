### Table of Contents:
0. [Introduction](https://github.com/lucashu1/CDN-RecSys/wiki#introduction)
1. [Playing Catch-Up](https://github.com/lucashu1/CDN-RecSys/wiki#1-playing-catch-up)
2. [Data Preprocessing](https://github.com/lucashu1/CDN-RecSys/wiki#2-data-preprocessing)
3. [Creating the Recommendation Models](https://github.com/lucashu1/CDN-RecSys/wiki#3-creating-the-recommendation-models)
4. [Model Evaluation](https://github.com/lucashu1/CDN-RecSys/wiki#4-model-evaluation)
5. [Results](https://github.com/lucashu1/CDN-RecSys/wiki#5-results)
6. [Next Steps](https://github.com/lucashu1/CDN-RecSys/wiki#6-next-steps)
7. [Conclusion](https://github.com/lucashu1/CDN-RecSys/wiki#conclusion)

## Introduction
This summer, I was lucky enough to be invited to [Tsinghua University's Network and Big Data Technology R&D Center](http://cdn.riit.tsinghua.edu.cn) as a part of USC Viterbi's Research Abroad program. During my six weeks in China, I spent the weekdays working full-time in the lab, and the weekends exploring places like the Great Wall at Huanghuacheng, the Terracotta Army Museum in Xi'an, and the Panda Research Base in Chengdu -- all the while having some of the best food of my life.

As the name of the research lab would suggest, much of the work in the lab was focused on topics like computer networking (including content delivery networks, or CDNs) and distributed computing, with some additional work in data science and machine learning. While our research projects during the six-week program didn't necessarily have to fall within one of these categories, it made sense to me when my partner/host student suggested that I try my hand at creating a recommender system for CDN selection as my project. (The motivation was that although there were relatively few CDN providers about a decade ago, there are so many content delivery services out there now that it can be hard for companies to know which one(s) to pick.)

My partner said that he knew some researchers over in Nanjing that might have a dataset that could be applied to this problem, and that he would try to get it to me within a few weeks. Even though I knew practically nothing about CDNs going into the project, I decided that it still seemed like a pretty exciting problem to work on.

The next section describes the process that followed.

## CDN-RecSys Project

### 1. Playing Catch-Up

My first step, knowing almost nothing about recommender systems except for the 30-minutes worth of knowledge I had forgotten and then re-accumulated from Andrew Ng's Coursera video lectures on Machine Learning, was to go through some basic tutorials to learn the foundations of recommender systems, and then to do a whole lot of literature review to get myself up-to-speed on the current state of recommender systems research.

All in all, my learning progression went something like this:
1. Introductory online tutorials/lectures: e.g. [Andrew Ng's recommender system lectures](https://www.youtube.com/watch?v=giIXNoiqO_U), [Mining of Massive Datasets: Recommender Systems Series](https://www.youtube.com/watch?v=1JRrCEgiyHM)
2. Older academic papers: e.g. [Amazon: Item-to-Item Collaborative Filtering (2003)](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf), [Hu, et. al: Collaborative Filtering for Implicit Feedback Datasets (2008)](http://yifanhu.net/PUB/cf.pdf)
3. More recent blog posts: e.g. [Explicit Matrix Factorization: ALS, SGD, and All That Jazz](https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea), [Ethan Rosenthal: Intro to Implicit Matrix Factorization](http://blog.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/)
4. More recent academic papers: e.g. [Steffen Rendle: Factorization Machines (2010)](http://www.algo.uni-konstanz.de/members/rendle/pdf/Rendle2010FM.pdf), [Maciej Kula: Metadata Embeddings for User and Item Cold-Start Recommendations (2015)](https://arxiv.org/pdf/1507.08439.pdf)

As I gradually made my way through these materials, I learned about basic concepts like the distinction between content-based and collaborative-filtering recommender systems, and some older ideas on how to generate recommendations, such as k-nearest-neighbors. Eventually, I learned about more modern recommendation methods, such as matrix factorization and then factorization machines, as well as some ways to combine content-based and collaborative-filtering models into a hybrid approach. Along the way, I wrote some basic Python code to practice these ideas on the Movielens dataset, which helped me get some more hands-on RecSys experience.

It was certainly a lot to take in, but after a while, I felt that I had gained enough of an understanding of recommender systems to be able to apply these ideas to my own project.

### 2. Data Preprocessing

A couple weeks in, my partner let me know that he had received most of the data from the folks in Nanjing in .csv format (about 3GB total). It appeared to be data that was mostly collected from a web crawler that was visiting a bunch of Chinese websites and storing a bunch of information like what CDNs the websites used, how many bytes of text/images/video were on each webpage, and CDN-quality of service statistics like DNS resolution times, packet loss, etc. Since there's already plenty of research on measuring CDN performance, all I really needed were the ICP (internet content provider)-CDN interactions for implicit feedback, as well as some information on the ICPs and CDNs that I could turn into user and item features (to create a hybrid recommendation model).

With the help of Pandas (and Google Translate, since all the documentation for the data was in Chinese), I went through the important data tables and kept track of which ICPs used which CDNs (which I turned into an implicit feedback interactions matrix), along with some auxiliary information to use as my user/item content features. The features I kept were: 

* For ICPs (users): ICP name (unique ID), Industry, Total bytes of text, Total bytes of images, Total bytes of video
* For CDNs (items): CDN code (unique ID), CDN type (free/commercial), Number of CDN IP addresses

For the quantitative features, I divided each user/item into bins based on quartiles, and then used sklearn's DictVectorizer to turn all these features into one-hot encodings. Each ICP/CDN would be represented by an n-dimensional binary vector indicating which features were applicable to that ICP/CDN.

In the end, I had an interactions matrix (with ICPs as rows and CDNs as columns), an array of one-hot feature vectors for the ICPs, and another array of one-hot feature vectors for the CDNs. I was now ready to start building my recommendation model.

### 3. Creating the Recommendation Models

* Pure CF
* Hybrid models

### 4. Model Evaluation

* Train-test split
* Metrics: p@5 (because 5 is less than 6 but bigger than 4), AUC
* Compare models
* Warm-cold split

### 5. Results

* Meaningful learning in warm-start and cold-start scenarios
* ICP features were valuable
* CDN features need some tweaking

### 6. Next Steps

* Extra feature engineering
* Try different models (deep learning?)
* **More data**

## Conclusion

Promising first-step! Lots more work to be done.
