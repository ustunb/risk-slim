risk-slim
========

risk-slim is a machine learning method to fit simple customized risk scores in python. 

#### Background 

Risk scores let users make quick risk predictions by adding and subtracting a few small numbers (see e.g., 500 + medical risk scores at [mdcalc.com](https://www.mdcalc.com/)). 

Here is a risk score for ICU risk prediction from our [paper](http://www.berkustun.com/docs/ustun_2017_optimized_risk_scores.pdf). 

<div>
<p align="center">
<img src="https://github.com/ustunb/risk-slim/blob/master/docs/images/risk_score_seizure.png" width="480" height="360" border="0"/>
</p>
</div>

#### Video

<p align="center">
	<a href="http://www.youtube.com/watch?feature=player_embedded&v=WQDVejk17Aw" target="_blank">
		<img src="http://img.youtube.com/vi/WQDVejk17Aw/0.jpg" alt="RiskSLIM KDD" width="480" height="360" border="10" />
	</a>
</p>
 

#### Reference

If you use risk-slim in your research, we would appreciate a citation to the following paper ([bibtex](/docs/references/ustun2019riskslim.bib)!

<a href="http://jmlr.org/papers/v20/18-615.html" target="_blank">Learning Optimized Risk Scores</a> <br>
Berk Ustun and Cynthia Rudin<br>
Journal of Machine Learning Research, 2019.

## Installation

Run the following snippet in a Unix terminal to install risk-slim and complete a test run.  

```
git clone https://github.com/ustunb/risk-slim
cd risk-slim
pip install -e . 		# install in editable mode  
bash batch/job_template.sh 	# batch run
```

### Requirements

risk-slim requires Python 3.5+ and CPLEX 12.6+. For instructions on how to download and install, click [here](/docs/cplex_instructions.md). 



## Contributing

I'm planning to pick up development again in Fall 2020. I can definitely use a hand! If you are interested in contributing, please reach out!  

Here's the current development roadmap:

- [sci-kit learn interface](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator)
- support for open source solver in [python-mip](https://github.com/coin-or/python-mip)
- basic reporting tools (roc curves, calibration plots, model reports)
- documentation
- pip