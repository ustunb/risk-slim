"""Risk scores results, measures, and reports."""
import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import write_image

from riskslim.utils import print_model


class RiskScores:
    """Risk scores (rho), derived metrics, and reports."""

    def __init__(self, estimator, X, y, cv=None):

        if not isinstance(estimator, list) and not estimator.fitted:
            # Ensure single model if fit
            raise ValueError("RiskScores expects a fit RiskSLIM input.")

        # Unpack references to RiskSLIM arrays
        self._X = X
        self._y = y
        self.estimator = estimator
        self.rho = estimator.rho
        self._coef_set = estimator.coef_set
        self._variable_names = estimator.coef_set.variable_names
        self._outcome_name = estimator.outcome_name

        self.cv = cv

        # Table
        self._table = print_model(
            self.rho,
            self._variable_names,
            self._outcome_name,
            show_omitted_variables=False,
            return_only=True
        )
        self._preprare_table()

        # Performance measures
        self.proba = estimator.predict_proba(self._X)
        self.proba_true = None
        self.proba_pred = None
        self.fpr = None
        self.tpr = None


    def __str__(self):
        """Scores string."""
        return str(self._table)


    def __repr__(self):
        return str(self._table)


    def print_coefs(self):
        """Print coefficient info."""
        print(self.coef_set)


    def compute_metrics(self, y, proba):
        """Computes calibration and ROC."""

        # Calibration curve
        prob_true, prob_pred = calibration_curve(y, proba, pos_label=1)
        prob_pred *= 100
        prob_true *= 100

        # ROC curve
        fpr, tpr, _ = roc_curve(y.reshape(-1), proba)

        return prob_pred, prob_true, fpr, tpr


    def _preprare_table(self):
        """Prepare arrays for plotly table."""

        # Non-zero coefficients
        inds = np.where(self.rho[1:] != 0)[0]

        if len(inds) == 0:
            raise ValueError('No non-zero coefficients.')

        self._table_names = np.array(self._variable_names)[inds+1].tolist()
        self._table_scores = self.rho[inds+1]

        self._table_names = [str(i+1) + '.    ' + n
                             for i, n in enumerate(self._table_names)]
        self._table_names.append('')

        self._table_scores = [str(int(s)) + ' point' if s == 1 else str(int(s)) + ' points'
                              for s in self._table_scores]
        self._table_scores.append('SCORE')

        self._table_last_col = ['   ...', *['+ ...']*(len(self._table_scores)-2)]
        self._table_last_col.append('= ...')



    def report(self, file_name=None, show=True):
        """Create a RiskSLIM report using plotly.

        Parameters
        ----------
        file_name : str
            Name of file and extension to save report to.
            Supported extensions include ".pdf" and ".html".
        show : bool, optional, default: True
            Calls fig.show() if True.
        """

        self.prob_pred, self.prob_true, self.fpr, self.tpr = \
            self.compute_metrics(self._y, self.proba)

        # Initalize subplots
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"colspan": 2, "type": "table"}, None],
                [{"colspan": 2, "type": "table"}, None],
                [{}, {}],
            ],
            row_heights=[.3, .2, .5],
            vertical_spacing=.05,
            subplot_titles=("Scores", "Predicted Risk", "Calibration", "ROC Curve")
        )


        # Table: risk scores
        fig.add_trace(
            go.Table(
                header=dict(
                    values=[''],
                    height=0
                ),
                cells=dict(
                    values=[
                        [*self._table_names],
                        [*self._table_scores],
                        [*self._table_last_col]
                    ],
                    align='left',
                ),
                columnwidth=[400, 150, 150]
            ),
            row=1,
            col=1
        )

        # Table: predicted risk
        _pred_risk = np.stack((
            np.arange(len(self.prob_pred)),
            [str(round(i, 1))+"%" for i in self.prob_pred.tolist()]
        )).T

        fig.add_trace(
            go.Table(
                header=dict(
                    values=[''],
                    height=0
                ),
                cells=dict(
                    values=[
                        ['Score', 'Risk'],
                        *_pred_risk
                    ],
                    align='left',
                ),
            ),
            row=2,
            col=1
        )

        # Folds
        if self.cv is not None:

            for _, test in self.cv.split(self._X):

                # Calibration
                proba = self.estimator.predict_proba(self._X[test])

                prob_pred, prob_true, fpr, tpr = self.compute_metrics(
                    self._y[test], proba
                )

                fig.add_trace(
                    go.Scattergl(
                        x=prob_pred,
                        y=prob_true,
                        mode='markers+lines',
                        line=dict(color='black', width=2),
                        opacity=.2,
                        name=""
                    ),
                    row=3,
                    col=1
                )

                # ROC
                fig.add_trace(
                    go.Scattergl(
                        x=fpr,
                        y=tpr,
                        mode='lines',
                        line=dict(color='black', width=2),
                        name="",
                        opacity=.2
                    ),
                    row=3,
                    col=2
                )


        # Calibration
        fig.add_trace(
            go.Scattergl(
                x=self.prob_pred,
                y=self.prob_true,
                mode='markers+lines',
                line=dict(color='black', width=2),
                name="Model"
            ),
            row=3,
            col=1
        )

        fig.add_trace(
            go.Scattergl(
                x=np.linspace(0, 100),
                y=np.linspace(0, 100),
                mode='lines',
                line=dict(color='black', dash='dash', width=2),
                opacity=.5,
                name="Ideal"
            ),
            row=3,
            col=1
        )


        # ROC curve
        fig.add_trace(
            go.Scattergl(
                x=self.fpr,
                y=self.tpr,
                mode='lines',
                line=dict(color='black', width=2),
                name="ROC"
            ),
            row=3,
            col=2
        )

        fig.add_trace(
            go.Scattergl(
                x=np.linspace(0, 1),
                y=np.linspace(0, 1),
                mode='lines',
                line=dict(color='black', dash='dash', width=2),
                opacity=.5,
                name="Ideal"
            ),
            row=3,
            col=2
        )

        fig.update_layout(
            # General
            title_text="RiskSLIM Report",
            autosize=False,
            width=800,
            height=800,
            showlegend=False,
            template='simple_white',
            # Calibration axes
            yaxis1=dict(
                tickmode='array',
                tickvals=np.arange(0, 120, 20),
                ticktext=[str(i) + '%' for i in np.arange(0, 120, 20)],
                title='Observed Risk'
            ),
            xaxis1=dict(
                tickmode='array',
                tickvals=np.arange(0, 120, 20),
                ticktext=[str(i) + '%' for i in np.arange(0, 120, 20)],
                title='Predicted Risk'
            ),
            xaxis2=dict(
                range=(-.05, 1),
                tickmode='array',
                tickvals=np.linspace(0, 1, 6),
                title='False Positive Rate'

            ),
            yaxis2=dict(
                tickmode='array',
                tickvals=np.linspace(0, 1, 6),
                title='True Positive Rate'
            )
        )

        if file_name is not None and file_name.endswith('pdf'):
            # Save as pdf
            write_image(fig, file_name, format='pdf')
        elif file_name is not None and file_name.endswith('html'):
            # Save as html
            with open(file_name, 'w') as f:
                f.write(fig.to_html())
        elif file_name is not None:
            raise ValueError("Unsupported file extension. Use \".pdf\" or \".html\".")

        if show:
            fig.show()
