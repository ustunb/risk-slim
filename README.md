risk-slim
========

``risk-slim`` is a Python package to train data-driven *risk scores*. These are simple classification models that let users quickly assess risk by adding, subtracting and multiplying a few small numbers.

**NOTE: THIS PACKAGE IS CURRENTLY WORKING BUT UNDER ACTIVE DEVELOPMENT. There may be bugs, and the internal code may change substantially with each commit.** 



## Installation
  
Run the following snippet to install ``risk-slim`` in a Mac/Unix environment, and complete a test run.  

```
git clone https://github.com/ustunb/risk-slim
cd risk-slim
pip install -e . #install in editable mode  
bash batch/job_template.sh #batch run
```

### Requirements

``risk-slim`` was developed using Python 2.7.11 and CPLEX 12.6. It may work with older versions of Python and CPLEX. However this has not been tested and will not be supported.


### CPLEX 

CPLEX is cross-platform commercial optimization tool with a Python API. It is freely available to students and faculty members at accredited institutions as part of the IBM Academic Initiative. 

To get CPLEX:

1. Sign up for the [IBM Academic Initiative](https://developer.ibm.com/academic/). Note that it may take up to a week to obtain approval.
2. Download *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://ibm.onthehub.com/WebStore/OfferingDetails.aspx?o=6fcc1096-7169-e611-9420-b8ca3a5db7a1)
3. Install the file on your computer. Note mac/unix users will [need to install a .bin file](http://www-01.ibm.com/support/docview.wss?uid=swg21444285).
4. Setup the CPLEX Python API [as described here here](http://www.ibm.com/support/knowledgecenter/SSSA5P_12.6.3/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems installing CPLEX, check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059). 

## Development Timeline

- ~~Simplify Installation~~ 
- ~~Convenience functions for batch computing~~
- Refactoring/Specs/Unit Tests/Documentation
- Helper Functions for Performance Analysis
- OO Interface for Risk Score Models
- Compatability with [sci-kit learn](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator)
 

## Citation 

If you use ``risk-slim`` for in your research, please cite [our paper](https://arxiv.org/abs/1610.00168)!  
     
```
@inproceedings{ustun2017kdd,
	Author = {Ustun, Berk and Rudin, Cynthia},
	Booktitle = {Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
	Organization = {ACM},
	Title = {{Optimized Risk Scores}},
	Year = {2017}}
}
```




