risk-slim
========

**This package is currently under active development (should be up and running by March 2017).** 

risk-slim is a package to create optimized risk scores in Python using the CPLEX MIP solver.

*Risk scores* are simple classification models to assess risk by adding, subtracting and multiplying a few small numbers.



## Requirements

``risk-slim`` was developed using Python 2.7.11 and CPLEX 12.6.3. It may work with other versions of Python and/or CPLEX, but this has not been tested and will not be supported in future releases.

### Obtaining and Installing CPLEX 

*CPLEX* is cross-platform commercial optimization tool that can be called from Python. It is freely available to students and faculty members at accredited institutions as part of the IBM Academic Initiative. To get CPLEX:

1. Sign up for the [IBM Academic Initiative](https://developer.ibm.com/academic/). Note that it may take up to a week to obtain approval.
2. Download *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://ibm.onthehub.com/WebStore/OfferingDetails.aspx?o=6fcc1096-7169-e611-9420-b8ca3a5db7a1)
3. Install the file on your computer. Note mac/unix users will [need to install a .bin file](http://www-01.ibm.com/support/docview.wss?uid=swg21444285).
4. Setup the CPLEX Python API [as described here here](http://www.ibm.com/support/knowledgecenter/SSSA5P_12.6.3/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

Please check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059) if you have problems installing CPLEX.

## References 

If you use RiskSLIM for academic research, please cite [our paper](https://arxiv.org/abs/1610.00168)!  
     
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

