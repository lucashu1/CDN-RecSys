# CDN-RecSys: A Network That Works

An implementation of the LightFM hybrid recommender system for suggesting CDN (content delivery network) 
providers to various websites. Includes examples of using the LightFM to create both hybrid and pure collaborative
filtering recommender systems, using scikit-optimize to run hyperparameter searches, and matplotlib to
visualize the models' learning curves.

(Created at Tsinghua University's Big Data Technology R&D Center as part of USC Viterbi's Research Abroad program.)

### Motivation

[Add later]

### Pre-Requisites

If you would like to run the notebook locally, you'll need:
* Python >= 2.7, preferably via [Anaconda](https://www.continuum.io/downloads)
* The [SciPy Stack](https://www.scipy.org/stackspec.html) (included in Anaconda builds)
* [LightFM RecSys Library](https://lyst.github.io/lightfm/docs/home.html)
* [Scikit-Optimize](https://scikit-optimize.github.io)

Then, go ahead and clone the repository, create a Jupyter session within the directory, and open
the .ipynb file.


### Files Included

* LightFM.ipynb
* User/Item - Index Mappings (pkl)
* User/Item Feature Vectors (pkl)
* Train-test, Warm-cold interaction splits (pkl)


### Built With

* [LightFM](https://lyst.github.io/lightfm/docs/home.html) - for creating and training the recommender models
* Pandas - for data pre-processing
* Scikit-optimize - for hyperparameter optimization
* Numpy/scipy - for everything matrix-related

### Supplementary Materials

* Project Wiki
* Project Slides (PPT)


### Acknowledgments

* Professor Yin Hao, Professor Yu Longqiang (Tsinghua University)
* Zha Cong: my host/partner student at Tsinghua
* Maciej Kula (Lyst): LightFM creator
* Ethan Rosenthal: for his [blog posts](http://blog.ethanrosenthal.com) on recommender systems
