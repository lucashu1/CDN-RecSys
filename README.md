# CDN-RecSys: A Network That Works

An application of the LightFM hybrid recommender system library for suggesting CDN (content delivery network) 
providers to various websites. Includes examples of using LightFM to create both hybrid and pure collaborative
filtering recommender systems, using scikit-optimize to run hyperparameter searches, and matplotlib to
visualize the models' learning curves.

(Created at Tsinghua University's Big Data Technology R&D Center as part of USC Viterbi's Research Abroad program.)

### Motivation

(Coming Soon)

### Prerequisites

If you would like to run the notebook locally, you'll need:
* Python >= 2.7, preferably via [Anaconda](https://www.continuum.io/downloads)
* The [SciPy Stack](https://www.scipy.org/stackspec.html) (included in Anaconda builds)
* [LightFM](https://lyst.github.io/lightfm/docs/home.html) Recommender System Library (available via `pip`)
* [Scikit-Optimize](https://scikit-optimize.github.io) (available via `pip`)

Then, clone this repository, create a Jupyter session within the repo directory, and open
the LightFM.ipynb file.

### Files Included

* **LightFM.ipynb** - Main recommender system script
* **20170629-interactions-mappings.pkl** - (interactions, iidx_to_cdn, cdn_to_iidx, uidx_to_icp, icp_to_uidx) tuple
  * Interactions: rows = users (ICPs), cols = items (CDNs), nonzero entries = interactions (CSR-sparse matrix)
  * icp_to_uidx: ICP name (URL/domain) to user index mapping (Python dict)
  * uidx_to_icp: User index to ICP name (URL/domain) mapping (Python dict)
  * cdn_to_iidx: CDN code (3-digit) to item index mapping (Python dict)
  * iidx_to_cdn: Item index to CDN code (3-digit) mapping (Python dict)
* **20170703-cdn-feature-vectors.pkl** - cdn_features array, rows = one-hot feature vectors (CSR-sparse matrix)
* **20170703-icp-feature-vectors.pkl** - icp_features array, rows = one-hot feature vectors (CSR-sparse matrix)
* **20170705-train-test.pkl** - (train, test) matrices tuple (CSR-sparse matrices)
* **20170705-warm-cold.pkl** - (test_warm, test_cold) matrices tuple (CSR-sparse matrices)

### Built With

* [LightFM](https://lyst.github.io/lightfm/docs/home.html) - Recommender System Library
* [Pandas](https://pandas.pydata.org) - Data Preprocessing
* [Scikit-optimize](https://scikit-optimize.github.io) - Hyperparameter Optimization
* [Scikit-learn](http://scikit-learn.org/stable/) - For feature vectorization
* [Numpy](http://www.numpy.org)/[Scipy](https://www.scipy.org/scipylib/index.html) - For everything matrix-related

### Supplementary Materials

* Project Wiki (link coming soon)
* [Project Slides (PPT)](https://drive.google.com/open?id=0B9a6HGclbze9SW04V0h3dzVFaXM)

### Acknowledgments

* Professor Yin Hao, Professor Yu Longqiang (Tsinghua University)
* Zha Cong - My host/partner student at Tsinghua University
* Maciej Kula (Lyst) - LightFM Author
* Ethan Rosenthal - For his [blog posts](http://blog.ethanrosenthal.com) on recommender systems
