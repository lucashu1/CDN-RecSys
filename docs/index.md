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

As the name of the research lab would suggest, much of the work in the lab was focused on topics like computer networking (including content delivery networks, or CDNs) and distributed computing, with some additional work in data science and machine learning. The people in the lab told me that my research project during the six-week program didn't necessarily have to fall within one of these categories. However, given the lab's existing research focuses, it seemed to make sense to me when my partner/host student suggested that I try my hand at creating a recommender system for CDN selection as my project, by viewing internet content providers (ICPs, i.e. websites) as users and CDN providers as items. 

The basic motivation was that although there were relatively few CDN providers about a decade ago, there are so many content delivery services out there now that it can be hard for ICPs to know which one(s) to pick. Perhaps by finding similar websites and seeing which CDNs they are currently using, we would be able to generate meaningful recommendations for which CDN the website in question should use.

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

I started by learning about basic concepts like the distinction between content-based (CB) and collaborative-filtering (CF) recommender systems, and some older ideas on how to generate recommendations, such as the k-nearest-neighbors (KNN) algorithm. As I gradually got more comfortable with the basics, I began researching more modern recommendation methods, such as matrix factorization (MF) and then factorization machines, as well as some ways to combine content-based and collaborative-filtering models into a hybrid approach. Along the way, I wrote some basic Python code to practice these ideas on the Movielens dataset, which helped me get some more hands-on RecSys experience.

It was certainly a lot to take in, but after a while, I felt that I had gained enough of an understanding of recommender systems to be able to apply these ideas to my own project.

### 2. Data Preprocessing

A couple weeks in, my partner let me know that he had received most of the data from the folks in Nanjing in .csv format (about 3GB total). It appeared to be data that was mostly collected from a web crawler that was visiting a bunch of Chinese websites and storing information like what CDNs the websites used, how many bytes of text/images/video were on each webpage, CDN quality of service statistics like DNS resolution times, packet loss, etc. Since there's already plenty of research on measuring CDN performance, I figured I would leave the quality of service statistics out for now. All I really needed were the ICP (internet content provider)-CDN interactions for implicit feedback, as well as some information on the ICPs and CDNs that I could turn into user and item features (to create a hybrid recommendation model).

<p align="center">
<img src="https://github.com/lucashu1/CDN-RecSys/blob/master/dataset_example.png?raw=true" width="70%" align="middle" alt="Dataset Example Screenshot"/>
</p>

With the help of [Pandas](https://pandas.pydata.org) (and Google Translate, since all the documentation for the data was in Chinese), I went through the important data tables and kept track of which ICPs used which CDNs (which I then turned into an implicit feedback interactions matrix), along with some auxiliary information to use as my user/item content features. The features I kept were: 

* For **ICPs** (users): ICP name (unique ID), Industry, Total bytes of text, Total bytes of images, Total bytes of video
* For **CDNs** (items): CDN code (unique ID), CDN type (free/commercial), Number of CDN IP addresses

For the quantitative features (bytes of web content and number of IP addresses), I divided all the ICPs/CDNs into bins based on quartiles, and then used sklearn's DictVectorizer to turn all these features into one-hot encodings. Each ICP/CDN would be represented by an *n*-dimensional binary vector indicating which features were applicable to that ICP/CDN.

In the end, I had an interactions matrix (with ICPs as rows and CDNs as columns), an array of one-hot feature vectors for the ICPs, and another array of one-hot feature vectors for the CDNs. With these preprocessed arrays set up, I was now ready to start building my recommendation model.

### 3. Creating the Recommendation Models

I knew that having to write the code for the recommender systems by hand would have been a huge pain. Thankfully, Maciej Kula, the author of the LightFM hybrid recommendation algorithm whose paper I linked earlier, created a [LightFM Python library](https://github.com/lyst/lightfm) with implementations of his recommendation algorithm built in. Essentially, what the LightFM algorithm does is instead of just learning latent vector representations for each user/item (as is the case in traditional MF-based collaborative filtering), LightFM learns latent vector representations for user/item *metadata* (i.e. features/tags). (You can include unique IDs in the metadata as well if you want to still have some unique component for each user/item.) Then, you get the total representation for a user/item by summing over each of its features' latent vectors. That way, you can augment the core collaborative filtering component with some content-based learning via the user/item feature vectors, making this a hybrid recommendation model. You can think of this like word2vec, but for user/item features instead of for words/phrases.

For example, if you wanted to get a representation for me, Lucas Hu, you'd first learn the latent vectors for "Age: 19", "School: USC", "Gender: Male", "Name: Lucas Hu", and then add those up. In the case of this project, you'd get the representation for an ICP by summing up the latent vectors for its industry, text/image/video bytes data, and unique website name.

The library can also handle basic implicit feedback collaborative filtering models, if you'd prefer to leave out the user/item feature vectors and skip the content-based learning component.

After installing this library, all I really had to do to create my models was to import some classes from the `LightFM` library, create some `LightFM` model objects, and then pass my own data into the built-in `fit()` function. Tweaking stuff like the SGD learning rate, regularization parameters, etc. was as easy as adding in an extra parameter to the model constructor.

Since I wanted to see if adding the ICP and CDN features actually helped generate more useful recommendations, I went ahead and created 4 different recommendation models:

* Pure CF: **No features**
* Hybrid: **ICP features** only
* Hybrid: **CDN features** only
* Hybrid: **All** ICP/CDN features 

The next section will talk about how I evaluated each of these models to find out which one performed the best.

### 4. Model Evaluation

To create an unbiased sample of interactions with which to test the different recommendation models, I had randomly selected 20% of the interactions from the interactions matrix and moved them into a separate `test` array, keeping the remaining 80% in a `train` array (which, as the name suggests, was used to train each of the recommendation models).

I selected two main metrics for evaluating the performance of the models against the test set:
* Precision at k (**p@k**): the fraction of the top-*k* suggestions that turned out to be known positives. (For my project, I picked k=5 because 5 is less than 6 but bigger than 4.)
* AUROC (**AUC** score): the probability that if we randomly draw a positive and a negative sample, our model will rank the positive sample above the negative one

For each of the 4 models, I then used [scikit-optimize](https://scikit-optimize.github.io) to run a smart hyperparameter search to get the best `test` p@5 possible. (Scikit-optimize's Bayesian optimization approach turned out to be way more efficient than a plain grid search.) I then compared the optimal p@5 scores between each model to see which model performed the best. This would also tell me how much of a performance boost I could get by adding each group of features (ICP or CDN) to the base model (pure CF).

Finally, I went back and analyzed the train-test split to see which `test` users were warm-start (i.e. already had some interactions in the training set) and which `test` users were cold-start (i.e. no interactions in the training set/completely new to the model), and then used this information to further split the `test` array into separate `test_warm` and `test_cold` interactions matrices. This would allow me to later see how the recommendations model did in both warm-start and cold-start scenarios. (Typically, cold-start scenarios are much more difficult for collaborative filtering models to tackle, since they rely heavily on user behavior to generate predictions.)

### 5. Results

When all was said and done, and the scikit-optimize commands had all finished doing their thing, here are the optimal overall test p@5 scores that I got for each model:

* No features (pure CF): 0.157105
* ICP features only: **0.158713** (top score)
* CDN features only: 0.157105
* All ICP/CDN features: 0.157641

There are a couple key takeaways here. First, notice that the CDN features provided **zero** performance boost to the pure CF model. In fact, when we remove the CDN features from the all-features model, we actually see a sizable **increase** in performance! That probably means that I didn't do a very good job on my part of selecting characteristic features for the CDNs, or simply that there wasn't much of a pattern at all within the CDN features, which would make them more empty noise than anything else.

It was reassuring to see, however, that adding the ICP features to the model did indeed produce a noticeable increase in overall performance from the base model. This means that the ICP features I selected were adding some valuable information to the model, and helping the model generate more meaningful CDN recommendations based on the website characteristics in a variety of test cases.

<p align="center">
<img src="https://github.com/lucashu1/CDN-RecSys/blob/master/icp_features_only.png?raw=true" alt="ICP Features Only Learning Curve" width="70%"/>
</p>

The biggest benefit to adding the ICP features, however, turned out to be in the **cold-start** scenario. After finding that the ICP-features-only model performed best out of the four models I tested, I went ahead and evaluated its performance more in-depth using the separate warm-start and cold-start test interaction matrices I had mentioned earlier. I found that in the warm-start case, adding the ICP features to the base model produced only a tiny increase in p@5 performance (from 0.159833 to 0.160669). In the cold-start scenarios, though, adding the ICP features increased the model's p@5 from 0.152239 to 0.155224, for a more significant performance boost of 2%. (I know, still a pretty small boost -- but better than nothing, right?)

This result was somewhat expected because in the cold-start case, in which the recommender system doesn't know anything about the user's behavior, all the collaborative filtering algorithm can really do is simply suggest the most popular items to that user, which may or may not result in helpful recommendations. If we factor in the user's metadata/features, however, we allow the model to use those features to find similar users, and then use those similar users' purchasing decisions to generate more personalized recommendations for the new user.

This means that even if you're a new user, the model will be able to do more than simply show you a New York Times best seller list. (I wonder what's topping "Hardcover Fiction" these days?)

### 6. Potential Next Steps

If I had more time, I would want to do some extra feature engineering to see if I could pull some meaningful CDN features from the raw data. Perhaps I just transformed the current data wrong; perhaps I need to consider completely different data altogether. Either way, it'd be nice to see some performance jump from adding in features to help characterize each CDN provider.

I would also want to see if different models could perform any better than LightFM did in this case. It's possible that creating a more customized CDN-recommendation model that takes CDN performance (e.g. latency, reliability, DNS resolve times, etc.) into account may perform better in the real world than my naive implicit feedback approach. Deep learning seems to all the rage these days, too, so maybe it would be possible to use deep learning to create a new recommendation model as [many others seem to have already done](http://bdsc.lab.uic.edu/docs/survey-critique-deep.pdf).

Finally, and most importantly, I would want to acquire some **more data** before turning this into a more finalized project, or even an academic paper. The current dataset included about 2,000 unique ICPs and only 40 unique CDNs: likely not enough information to be all that helpful in the real world. Since the researchers in Nanjing who collected the data that I used seemed to have used some sort of web crawler to navigate around the Chinese internet, perhaps it would be possible to do something similar for the worldwide web, and not just within China. The head professor of the lab, Prof. Yin Hao, seemed to think that collecting data from the U.S., which has many more CDNs services available, would produce a lot more meaningful results.

## Conclusion

I recognize that this project, as well as my own understanding of recommender systems, still has a ways to go -- but viewed in terms of a learning experience, keeping in mind that I knew almost nothing about recommender systems my first day in the program, I feel that it's been pretty amazing. (It's also exposed me to a lot of other really cool concepts like latent vector representations, which makes me want to learn a lot more about ideas like word2vec.) 

I'm still not completely sure if I'll continue to work on this project in the future; but either way, I think it's been great in terms of preparing me to work on more professional research projects. All in all, not a bad first step!