risk-slim
========

risk-slim is a machine learning method to fit simple customized risk scores in python. 

#### Background 

Risk scores let users make quick risk predictions by adding and subtracting a few small numbers (see e.g., 500 + medical risk scores at [mdcalc.com](https://www.mdcalc.com/)). 

Here is a risk score for ICU risk prediction from our [paper](http://www.berkustun.com/docs/ustun_2017_optimized_risk_scores.pdf). 

<div>
<p align="center">
<img src="https://github.com/ustunb/risk-slim/blob/master/images/risk_score_seizure.png" width="480" height="360" border="0"/>
</p>
</div>

#### Video

<p align="center">
	<a href="http://www.youtube.com/watch?feature=player_embedded&v=WQDVejk17Aw" target="_blank">
		<img src="http://img.youtube.com/vi/WQDVejk17Aw/0.jpg" alt="RiskSLIM KDD" width="480" height="360" border="10" />
	</a>
</p>
 

#### Reference

If you use risk-slim in your research, please cite our paper!

<a href="http://jmlr.org/papers/v20/18-615.html" target="_blank">Learning Optimized Risk Scores</a> <br>
Berk Ustun and Cynthia Rudin<br>
Journal of Machine Learning Research, 2019.

```
@article{ustun2019jmlr,
  author  = {Ustun, Berk and Rudin, Cynthia},
  title   = {{Learning Optimized Risk Scores}},
  journal = {{Journal of Machine Learning Research}},
  year    = {2019},
  volume  = {20},
  number  = {150},
  pages   = {1-75},
  url     = {http://jmlr.org/papers/v20/18-615.html}
}
```

## Installation

risk-slim requires Python 3.5+ and CPLEX 12.6+.

### Quick Install

Run the following snippet in a Unix terminal to install risk-slim and complete a test run.  

```
git clone https://github.com/ustunb/risk-slim
cd risk-slim
pip install -e . 		# install in editable mode  
bash batch/job_template.sh 	# batch run
```

### CPLEX

CPLEX is cross-platform optimization tool solver a Python API. It is free for students and faculty members at accredited institutions. To download CPLEX:

1. Register for [IBM OnTheHub](https://ur.us-south.cf.appdomain.cloud/a2mt/email-auth)
2. Download the *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://www-03.ibm.com/isc/esd/dswdown/searchPartNumber.wss?partNumber=CJ6BPML)
3. Install CPLEX Optimization Studio.
4. Setup the CPLEX Python API [as described here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems with CPLEX, please check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059). 

## Contributing

I'm planning to pick up development again in Fall 2020. I can definitely use a hand! If you are interested in contributing, please reach out to [berk@seas.harvard.edu](mailto:berk@seas.harvard.edu)! 

Here's the current development roadmap:

- [sci-kit learn interface](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator)
- support for open source solver and Gurobi via [python-mip](https://github.com/coin-or/python-mip)
- reporting tools (roc curves, calibration plots, model reports)
- documentation
- pip