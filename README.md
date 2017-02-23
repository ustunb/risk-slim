risk-slim
========

**THIS PACKAGE IS UNDER ACTIVE DEVELOPMENT. It will be finalized in March 2017.** 

risk-slim is a Python package to build data-driven *risk scores*. These are simple classification models to assess risk by adding, subtracting and multiplying a few small numbers.

## Installation 

``risk-slim`` was developed using Python 2.7.11 and CPLEX 12.6.3. It should work with older versions of Python and CPLEX, however this has not been tested and will not be supported.

### CPLEX 

*CPLEX* is cross-platform commercial optimization tool that can be called from Python. It is freely available to students and faculty members at accredited institutions as part of the IBM Academic Initiative. To get CPLEX:

1. Sign up for the [IBM Academic Initiative](https://developer.ibm.com/academic/). Note that it may take up to a week to obtain approval.
2. Download *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://ibm.onthehub.com/WebStore/OfferingDetails.aspx?o=6fcc1096-7169-e611-9420-b8ca3a5db7a1)
3. Install the file on your computer. Note mac/unix users will [need to install a .bin file](http://www-01.ibm.com/support/docview.wss?uid=swg21444285).
4. Setup the CPLEX Python API [as described here here](http://www.ibm.com/support/knowledgecenter/SSSA5P_12.6.3/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

Please check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059) if you have problems installing CPLEX.

### Fast Data-Related Computation

Performance can be substantially through special functions to compute the loss. To use these functions:
 
``python build_cython_loss_functions.py build_ext --inplace``

## Development Plan

##### Short Term:

- Specs, unit tests, documentation
- Examples
- Convenience functions to visualize performance (ROC curves, Calibation Diagrams)
- Convenience functions for batch computing on EC2
- Installation via pip

##### Long Term:

- Ability to use an open-source MIP solver [SYMPHONY IP solver](https://projects.coin-or.org/SYMPHONY) 
- Full [sci-kit learn compatibility](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator)

If you are interested in helping out, shoot me an e-mail.

## References 

If you use RiskSLIM for research, please cite [our paper](https://arxiv.org/abs/1610.00168)!  
     
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

