risk-slim
========

risk-slim is a machine learning method to learn customized risk scores from data.

### Background 

Risk scores are simple models that let users make quick risk predictions by adding and subtracting a few small numbers (see 500 + medical risk scores at [mdcalc.com](https://www.mdcalc.com/) or the [mdcalc iOS app](https://itunes.apple.com/us/app/mdcalc-medical-calculators-clinical-scores/id1001640662?ls=1&mt=8)).

#### Video

<p align="center">
	<a href="http://www.youtube.com/watch?feature=player_embedded&v=WQDVejk17Aw" target="_blank">
		<img src="http://img.youtube.com/vi/WQDVejk17Aw/0.jpg" alt="RiskSLIM KDD" width="480" height="360" border="10" />
	</a>
</p>

#### Customized Risk Score for ICU seizure prediction 

Here is a risk score for ICU risk prediction from our [paper](http://www.berkustun.com/docs/ustun_2017_optimized_risk_scores.pdf). 

<div>
<p align="center">
<img src="https://github.com/ustunb/risk-slim/blob/master/images/risk_score_seizure.png" width="480" height="360" border="0"/>
</p>
</div>

#### References 

If you use risk-slim in your research, please cite one of the following papers:

1. <a href="http://jmlr.org/papers/v20/18-615.html" target="_blank">Learning Optimized Risk Scores</a> <br>
Berk Ustun and Cynthia Rudin<br>
Journal of Machine Learning Research (JMLR), 2019.

2. <a href="http://www.berkustun.com/docs/ustun_2017_optimized_risk_scores.pdf" target="_blank">Optimized Risk Scores</a> <br>
Berk Ustun and Cynthia Rudin<br>
23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2017.


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

@inproceedings{ustun2017kdd,
  author  = {Ustun, Berk and Rudin, Cynthia},
  booktitle = {Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  organization = {ACM},
  title = {{Optimized Risk Scores}},
  year = {2017}
}
```

## Package Details

### Installation
  
Run the following snippet in a Unix terminal to install risk-slim and complete a test run.  

```
git clone https://github.com/ustunb/risk-slim
cd risk-slim
pip install -e . 		# install in editable mode  
bash batch/job_template.sh 	# batch run
```

### Requirements

- Python 3.5+ 
- CPLEX 12.6+
 
The code may work with older versions of Python and CPLEX, but this will not be supported. 

#### CPLEX 

CPLEX is cross-platform commercial optimization tool with a Python API. It is free for students and faculty members at accredited institutions. To get CPLEX:

1. Register for [IBM OnTheHub](https://ibm.onthehub.com/WebStore/Account/VerifyEmailDomain.aspx)
2. Download the *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://ibm.onthehub.com/WebStore/ProductSearchOfferingList.aspx?srch=CPLEX)
3. Install CPLEX Optimization Studio.
4. Setup the CPLEX Python API [as described here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems installing CPLEX, check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059). 

## Development Roadmap

**If you are interested in contributing, please reach out to [berk@seas.harvard.edu](mailto:berk@seas.harvard.edu)!**

- ~~simplify installation~~ 
- ~~convenience functions for batch computing~~
- ~~refactoring package for future development~~
- [sci-kit learn](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator) API
- reporting tools (roc curves, calibration plots, model reports)
- support for open-source solver
- documentation



