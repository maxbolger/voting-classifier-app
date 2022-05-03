import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score

def viz(data, model1, model2, model3, feat1, feat2, weights):
  """
  Creates a decision boundary chart based on the user's selections

        Parameters:
              data (pandas.DataFrame): Palmer Penguins Data
              model1 (class): first selected model
              model2 (class): second selected model
              model3 (class): third selected model
              feat1 (str): first selected feature
              feat2 (str): second selected feature
              weights (list): list of selected weights
  """
  np.random.seed(125)
  X = np.array(data[[feat1, feat2]])    
  ss = StandardScaler()
  X = ss.fit_transform(X)
  x0 = np.array([i[0] for i in X])
  x1 = np.array([i[1] for i in X])
  y = np.array(data.species)
  le = LabelEncoder()
  y = le.fit_transform(y)

  PAD = 1.0
  x0_min, x0_max = np.round(x0.min())-PAD, np.round(x0.max()+PAD)
  x1_min, x1_max = np.round(x1.min())-PAD, np.round(x1.max()+PAD)

  H = 0.1
  x0_axis_range = np.arange(x0_min, x0_max, H)
  x1_axis_range = np.arange(x1_min, x1_max, H)

  xx0, xx1 = np.meshgrid(x0_axis_range, x1_axis_range)

  xx = np.reshape(np.stack((xx0.ravel(), xx1.ravel()), axis=1), (-1, 2))

  PROB_DOT_SCALE = 20
  PROB_DOT_SCALE_POWER = 4
  TRUE_DOT_SIZE = 50

  green = '#327374'
  purple = '#b764c5'
  orange = '#ee722e'

  colormap = np.array([orange, purple, green])

  vot = VotingClassifier(estimators=[
                                    ('clf1', model1[0]),
                                    ('clf2', model2[0]),
                                    ('clf3', model3[0])
                                    ],
                        voting='soft',
                        weights=[weights[0], weights[1], weights[2]]
                        )

  clfs = [model1[0], model2[0], model3[0], vot]
  labs = [model1[1], model2[1], model3[1], 'Voting Classifier']

  metrics_df = pd.DataFrame()

  fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True, constrained_layout=True)

  fig.supxlabel(f'{feat1} (scaled)', color='white')
  fig.supylabel(f'{feat2} (scaled)', color='white')

  ax1, ax2, ax3, ax4 = ax.flatten()

  for ax,clf,lab in zip(ax.flatten(), clfs, labs):

    ax.set_facecolor(color='#0E1117')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_title(lab, color='white')

    clf.fit(X, y)
    yy_hat = clf.predict(xx) 
    yy_prob = clf.predict_proba(xx) 
    yy_size = np.max(yy_prob, axis=1) 

    model = pd.DataFrame(
        {
          'accuracy': cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean(),
          'log_loss': abs(cross_val_score(clf, X, y, cv=5, scoring='neg_log_loss')).mean(),
          'precision': cross_val_score(clf, X, y, cv=5, scoring='precision_weighted').mean(),
          'recall': cross_val_score(clf, X, y, cv=5, scoring='recall_weighted').mean(),
          # 'roc_auc': cross_val_score(clf, X, y, cv=5, scoring='roc_auc_ovr_weighted').mean() # takes too long
        },
        index = [lab]
    )

    metrics_df = metrics_df.append(model)

    ax.scatter(
      xx[:,0], xx[:,1], c=colormap[yy_hat], alpha=0.4, 
      s=PROB_DOT_SCALE*yy_size**PROB_DOT_SCALE_POWER, linewidths=0
      )

    ax.contour(
      x0_axis_range, x1_axis_range,
      np.reshape(yy_hat, (xx0.shape[0], -1)),
      levels=3, linewidths=1, colors=[orange, purple, purple, green]
      )

    ax.scatter(x0, x1, c=colormap[y], s=TRUE_DOT_SIZE, zorder=3, linewidths=0.7, edgecolor='k')

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.set_yticks(np.arange(x1_min,x1_max, 1))
    ax.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xticks(ax.get_xticks()[1:-1])
    ax.set_yticks(np.arange(x1_min,x1_max, 1)[1:])
    ax.title.set_text(f'{lab}')

  legend_class = []
  for pingu_class, color in zip(['Chinstrap', 'Gentoo','Adelie'], [purple, green, orange]):
      legend_class.append(
          Line2D([0], [0], marker='o', label=pingu_class,ls='None',
          markerfacecolor=color, markersize=10, markeredgecolor='k', markeredgewidth=0.7)
        )


  prob_values = [0.4, 0.6, 0.8, 1.0]
  legend_prob = []
  for prob in prob_values:
      legend_prob.append(
          Line2D([0], [0], marker='o', label=prob, ls='None', alpha=0.8, markerfacecolor='white', 
          markersize=np.sqrt(PROB_DOT_SCALE*prob**PROB_DOT_SCALE_POWER)*2, markeredgecolor='k', markeredgewidth=0)
        )


  legend1 = fig.legend(
      handles=legend_class, labelcolor='white',loc='center', 
      bbox_to_anchor=(1.07, 0.35), frameon=False, title='Class'
    )

  plt.setp(legend1.get_title(), color='white')

  legend2 = fig.legend(
      handles=legend_prob, loc='center', labelcolor='white',
      bbox_to_anchor=(1.065, 0.65), frameon=False, title='Probability'
    )
  plt.setp(legend2.get_title(), color='white')
  fig.add_artist(legend1)

  return fig, metrics_df
