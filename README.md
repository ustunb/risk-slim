risk-slim
========

**NOTE: THIS PACKAGE IS CURRENTLY UNDER ACTIVE DEVELOPMENT. The code works but may be buggy, and may change substantially with each commit.** 

``risk-slim`` is a package to create simple data-driven *risk scores* in Python. These are simple classification models that let users quickly assess risk by adding, subtracting and multiplying a few small numbers.


## Installation 

``risk-slim`` was developed using Python 2.7 and CPLEX 12.6. It should work with older versions of Python and CPLEX. However this has not been tested and will not be supported.

### CPLEX 

CPLEX is cross-platform commercial optimization tool with a Python API. It is freely available to students and faculty members at accredited institutions as part of the IBM Academic Initiative. 

To get CPLEX:

1. Sign up for the [IBM Academic Initiative](https://developer.ibm.com/academic/). Note that it may take up to a week to obtain approval.
2. Download *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://ibm.onthehub.com/WebStore/OfferingDetails.aspx?o=6fcc1096-7169-e611-9420-b8ca3a5db7a1)
3. Install the file on your computer. Note mac/unix users will [need to install a .bin file](http://www-01.ibm.com/support/docview.wss?uid=swg21444285).
4. Setup the CPLEX Python API [as described here here](http://www.ibm.com/support/knowledgecenter/SSSA5P_12.6.3/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems installing CPLEX, check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059). 

## Development Timeline

##### Short Term:

- ~~Simplify Installation~~ 
- ~~Convenience functions for batch computing~~
- Convenience functions to assess performance (ROC curves, calibration diagrams)
- Examples / Jupyter Notebooks
- Specs/Unit Tests/Documentation

##### Long Term:

- Ability to use an open-source MIP solver ([SYMPHONY](https://projects.coin-or.org/SYMPHONY)) 
- Compatability with [sci-kit learn](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator)


## Citation 

If you use ``risk-slim`` research, please cite [our paper](https://arxiv.org/abs/1610.00168)!  
     
```
@article{ustun2016learning,
  title={Learning Optimized Risk Scores on Large-Scale Datasets},
  author={Ustun, Berk and Rudin, Cynthia},
  journal={stat},
  volume={1050},
  pages={1},
  year={2016}
}
```

