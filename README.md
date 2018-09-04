# CDN-RecSys: A Network That Works

An application of the LightFM hybrid recommender system library for suggesting CDN (content delivery network) 
providers to various websites. Includes examples of using LightFM to create both hybrid and pure collaborative
filtering recommendation models, scikit-optimize to run hyperparameter efficient searches, and matplotlib to
visualize the models' learning curves. **Detailed write-up [here](https://github.com/lucashu1/CDN-RecSys/wiki)**.

(Created at Tsinghua University's Big Data Technology R&D Center as part of USC Viterbi's Research Overseas program.)

**If you use this repo for your work, please cite the corresponding DOI:**

[![DOI](https://zenodo.org/badge/111464159.svg)](https://zenodo.org/badge/latestdoi/111464159)

## Motivation

Content Delivery Networks (CDNs) are great for getting your digital content onto the devices of end-users
around the world in an efficient way; however, there are so many different CDN providers and services
available now that it can be hard for new internet content providers (ICPs) to know which service to pick.

Using data on Chinese ICPs from some researchers in Nanjing, China, I formulated CDN selection as a recommender systems problem, with ICPs as users and CDN providers as items. I first constructed an interactions matrix of existing CDN purchases, as well as feature vectors for both the ICPs and CDNs in the dataset. The result was a hybrid recommender system model that could suggest CDN providers to ICPs in both warm-start and cold-start scenarios.

Although I don't have permission to release the raw data (I've hidden the actual CDN provider names, along with most of the raw ICP/CDN data tables), this project may still serve as a valuable first step (or perhaps even just a proof of concept) toward future applications of recommender systems for CDN selection.

## Prerequisites

If you would like to run the main notebook locally, you'll need:
* Python >= 2.7, preferably via [Anaconda](https://www.continuum.io/downloads)
* The [SciPy Stack](https://www.scipy.org/stackspec.html) (included in Anaconda builds)
* [LightFM](https://lyst.github.io/lightfm/docs/home.html) Recommender System Library (available via `pip`)
* [Scikit-Optimize](https://scikit-optimize.github.io) (available via `pip`)

Then, clone this repository, create a Jupyter session within the repo directory, and open
the LightFM.ipynb file.

## Files Included

#### IPython Notebooks

* **LightFM.ipynb** - Main recommender system notebook
* **Read_Interactions.ipynb** - Interactions preprocessing (not runnable)
* **Read_ICP_Features.ipynb** - User (ICP) feature preprocessing (not runnable)
* **Read_CDN_Features.ipynb** - Item (CDN) feature preprocessing (not runnable)

#### Pickle Dumps
* **20170629-interactions-mappings.pkl** - (interactions, iidx_to_cdn, cdn_to_iidx, uidx_to_icp, icp_to_uidx) tuple
  * Interactions: rows = users (ICPs), cols = items (CDNs), nonzero entries = interactions (CSR-sparse matrix)
  * icp_to_uidx: ICP name to user index mapping (Python dict)
  * uidx_to_icp: User index to ICP name mapping (Python dict)
  * cdn_to_iidx: CDN code to item index mapping (Python dict)
  * iidx_to_cdn: Item index to CDN code mapping (Python dict)
* **20170703-cdn-feature-vectors.pkl** - cdn_features array. Rows = one-hot feature vectors (CSR-sparse matrix)
* **20170703-icp-feature-vectors.pkl** - icp_features array. Rows = one-hot feature vectors (CSR-sparse matrix)  
* **20170705-train-test.pkl** - (train, test) matrices tuple (CSR-sparse matrices)
* **20170705-warm-cold.pkl** - (test_warm, test_cold) matrices tuple (CSR-sparse matrices)
* **20170714-opt-models.pkl** - tuple of trained LightFM model objects
  * Order: (opt_model_all, opt_model_none, opt_model_icp, opt_model_cdn)
* **20170714-opt-hyperparams.pkl** - tuple of optimal hyperparameters for each model
  * Order: (opt_hyperparams_all, opt_hyperparams_none, opt_hyperparams_icp, opt_hyperparams_cdn)
  * Each element contains: (opt_epochs_[name], opt_lr_[name], opt_no_components_[name], opt_item_alpha_[name], opt_user_alpha_[name])

## Built With

* [LightFM](https://lyst.github.io/lightfm/docs/home.html) - Recommender System Library
* [Pandas](https://pandas.pydata.org) - Data Preprocessing
* [Scikit-optimize](https://scikit-optimize.github.io) - Hyperparameter Optimization
* [Scikit-learn](http://scikit-learn.org/stable/) - For feature vectorization
* [Numpy](http://www.numpy.org)/[Scipy](https://www.scipy.org/scipylib/index.html) - For everything matrix-related

## Additional Materials

* [Project Wiki](https://github.com/lucashu1/CDN-RecSys/wiki)
* [Project Slides (PPT)](https://drive.google.com/open?id=0B9a6HGclbze9SW04V0h3dzVFaXM)

## Acknowledgments

* Professor Yin Hao, Professor Yu Longqiang (Tsinghua University)
* Zha Cong - My host/partner student at Tsinghua University
* Mr. Feng Deng - Donor and creator of the USC Viterbi/Tsinghua Summer Research Program
* Maciej Kula (Lyst) - LightFM Author
* Ethan Rosenthal - For his [blog posts](http://blog.ethanrosenthal.com) on recommender systems
