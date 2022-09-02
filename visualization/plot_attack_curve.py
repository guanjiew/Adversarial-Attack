from collections import Counter, defaultdict

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
import numpy as np
import pandas as pd
import json
import os


import matplotlib.pyplot as plt

os.environ['PYTHONPATH'] = '/h/zycluke/Project/DESSAL'

line_config = dict(
    random_sample=dict(name='Random', type="scatter", mode='lines+markers',
                       line=dict(color='rgb(31, 119, 180)', dash='solid'),
                       marker=dict(color='rgb(31, 119, 180)', symbol='square', size=8)),
    entropy_sample=dict(name='MaxEntropy', type="scatter", mode='lines+markers',
                        line=dict(color='rgb(227, 119, 194)', dash='solid'),
                        marker=dict(color='rgb(227, 119, 194)', symbol='diamond', size=8)),
    margin_sample=dict(name='Margin', type="scatter", mode='lines+markers',
                       line=dict(color='rgb(44, 160, 44)', dash='solid'),
                       marker=dict(color='rgb(44, 160, 44)', symbol='x', size=8)),
    badge=dict(name='BADGE', type="scatter", mode='lines+markers', line=dict(color='rgb(148, 103, 189)', dash='solid'),
               marker=dict(color='rgb(148, 103, 189)', symbol='circle', size=8)),
    badge_greedy=dict(name='BADGE (deterministic)', type="scatter", mode='lines+markers',
                      line=dict(color='rgb(148, 103, 189)', dash='dash'),
                      marker=dict(color='rgb(148, 103, 189)', symbol='circle', size=8)),
    bald=dict(name='BALD', type="scatter", mode='lines+markers', line=dict(color='rgb(255, 127, 14)', dash='solid'),
              marker=dict(color='rgb(255, 127, 14)', symbol='cross', size=8)),
    batchbald=dict(name='BatchBALD', type="scatter", mode='lines+markers',
                   line=dict(color='rgb(140, 86, 75)', dash='solid'),
                   marker=dict(color='rgb(140, 86, 75)', symbol='hexagon', size=8)),
    coreset=dict(name='Coreset', type="scatter", mode='lines+markers',
                 line=dict(color='rgb(8, 191, 194)', dash='solid'),
                 marker=dict(color='rgb(8, 191, 194)', symbol='star-triangle-up', size=8)),

    PDAOPL=dict(name='DAO', type="scatter", mode='lines+markers', line=dict(color='rgb(214, 39, 40)', dash='solid'),
                marker=dict(color='rgb(214, 39, 40)', symbol='star', size=8)),
    DAOPL=dict(name='DAO (deterministic)', type="scatter", mode='lines+markers',
               line=dict(color='rgb(214, 39, 40)', dash='dash'),
               marker=dict(color='rgb(214, 39, 40)', symbol='star', size=8)),
    damping=dict(name='Damping', type="scatter", mode='lines+markers',
                 line=dict(color='rgb(31, 119, 180)', dash='solid'),
                 marker=dict(color='rgb(31, 119, 180)', symbol='square', size=8)),
    incremental=dict(name='Incremental Training (DAO)', type="scatter", mode='lines+markers',
                     line=dict(color='rgb(31, 119, 180)', dash='solid'),
                     marker=dict(color='rgb(31, 119, 180)', symbol='square', size=8)),
)

def get_data_uncertainty(data_list, use_median=False, use_max=False, scale=False):
    '''
    Get data uncertainty for shaded traces.

    Parameters
    ----------
    data_list: :py:class:`list` of :py:class:`list`
        The y values for several predictors.
    use_median: :py:class:`bool`
        Use Mean or Median. Default to Mean.
    use_max: :py:class:`bool`
        Use Standard Deviation or Max and Min Value. Default to Standard Deviation.

    Returns
    -------

    '''
    y = []
    y_err = []
    high = []
    low = []
    for data in zip(*data_list):
        if scale:
            data = np.array(data) * 100
        if use_median:
            mean = np.nanmedian(data)
        else:
            mean = np.nanmean(data)
        y.append(mean)

        if use_max:
            up = np.nanmax(data)
            down = np.nanmin(data)
            y_err.append((up - down) / 2)
            high.append(up)
            low.append(down)
        else:
            std = np.nanstd(data)
            y_err.append(std)
            high.append(mean + std)
            low.append(mean - std)

    low = low[::-1]
    return y, y_err, high, low


def get_line_trace(df, x_col, y_cols, name, color=None, use_median=False, use_max=False, use_err_bar=False,
                   dash='solid', showlegend=False, scale=False):
    '''
    Get plotly trace for accuracy plot.

    Parameters
    ----------


    Returns
    -------
    :py:class:`dictionary`
        Plotly trace.

    '''
    config = line_config[name]

    x = df[x_col].tolist()
    y_list = [df[y_col].tolist() for y_col in y_cols]

    x_rev = x[::-1]
    y, y_err, high, low = get_data_uncertainty(
        y_list, use_median=use_median, use_max=use_max, scale=scale)

    #     print(config['marker'])

    if use_err_bar:
        traces = [
            dict(
                x=x,
                y=y,
                type=config['type'],
                line=config['line'],
                name=config['name'],
                #                 legendgroup = config['name'],
                showlegend=showlegend,
                error_y=dict(
                    type='data',
                    symmetric=True,
                    array=y_err),
                text=['{0:.3f}±{1:.3f}'.format(a, b) for a, b in zip(y, y_err)]
            )]
    else:
        traces = [
            dict(
                x=x,
                y=y,
                type=config['type'],
                line=config['line'],
                name=config['name'],
                #                 legendgroup = config['name'],
                showlegend=showlegend,
                text=['{0:.3f}±{1:.3f}'.format(a, b)
                      for a, b in zip(y, y_err)],
                marker=config['marker'],
            ),
            dict(
                x=x + x_rev,
                y=high + low,
                type=config['type'],
                line=dict(color='rgba(255,255,255,0)'),
                name=config['name'] + '_fill',
                #                 legendgroup = config['name'],
                hoverinfo='skip',
                fill='toself',
                fillcolor=config['line']['color'].replace('rgb', 'rgba').replace(')', ',0.2)'),
                showlegend=False,
            ),
        ]

    return traces

def get_data_with_err(y_list, y_bar_list, scale=False):
    '''
    Get data uncertainty for shaded traces.

    Parameters
    ----------
    data_list: :py:class:`list` of :py:class:`list`
        The y values for several predictors.
    use_median: :py:class:`bool`
        Use Mean or Median. Default to Mean.
    use_max: :py:class:`bool`
        Use Standard Deviation or Max and Min Value. Default to Standard Deviation.

    Returns
    -------

    '''
    assert len(y_list) == len(y_bar_list)
    high = []
    low = []
    new_y_list = []
    new_y_bar_list = []
    for y, y_bar in zip(y_list, y_bar_list):
        if scale:
            y = y * 100
            y_bar = y_bar * 100
        high.append(y + y_bar)
        low.append(y - y_bar)
        new_y_list.append(y)
        new_y_bar_list.append(y_bar)

    low = low[::-1]
    return new_y_list, new_y_bar_list, high, low


def get_line_trace_with_err(x, y_list, y_bar_list, name, use_err_bar=False, showlegend=False, scale=False):
    '''
    Get plotly trace for accuracy plot.

    Parameters
    ----------


    Returns
    -------
    :py:class:`dictionary`
        Plotly trace.

    '''
    config = line_config[name]

    x_rev = x[::-1]
    y, y_err, high, low = get_data_with_err(y_list, y_bar_list, scale=scale)

    if use_err_bar:
        traces = [
            dict(
                x=x,
                y=y,
                type=config['type'],
                line=config['line'],
                name=config['name'],
                #                 legendgroup = config['name'],
                showlegend=showlegend,
                error_y=dict(
                    type='data',
                    symmetric=True,
                    array=y_err),
                text=['{0:.3f}±{1:.3f}'.format(a, b) for a, b in zip(y, y_err)]
            )]
    else:
        traces = [
            dict(
                x=x,
                y=y,
                type=config['type'],
                line=config['line'],
                name=config['name'],
                #                 legendgroup = config['name'],
                showlegend=showlegend,
                text=['{0:.3f}±{1:.3f}'.format(a, b)
                      for a, b in zip(y, y_err)],
                marker=config['marker'],
            ),
            dict(
                x=x + x_rev,
                y=high + low,
                type=config['type'],
                line=dict(color='rgba(255,255,255,0)'),
                name=config['name'] + '_fill',
                #                 legendgroup = config['name'],
                hoverinfo='skip',
                fill='toself',
                fillcolor=config['line']['color'].replace(
                    'rgb', 'rgba').replace(')', ',0.2)'),
                showlegend=False,
            ),
        ]

    return traces


def robustness_to_query_emnist():
    query_size = [50, 100, 250, 500]
    acc = dict(
        random_sample=[0.762, 0.762, 0.762, 0.762],
        PDAOPL=[0.787, 0.789, 0.783, 0.777],
        badge=[0.787, 0.787, 0.785, 0.785],
        coreset=[0.727, 0.737, 0.739, 0.739],
        bald=[0.723, 0.726, 0.722, 0.659],
        margin_sample=[0.786, 0.785, 0.783, 0.758],
        entropy_sample=[0.700, 0.700, 0.614, 0.418],
        random=[0.762, 0.762, 0.765, 0.763]
    )
    err = dict(
        random_sample=[0.005, 0.004, 0.005, 0.004],
        PDAOPL=[0.011, 0.006, 0.008, 0.006],
        badge=[0.007, 0.004, 0.003, 0.002],
        coreset=[0.008, 0.008, 0.013, 0.009],
        bald=[0.001, 0.012, 0.008, 0.033],
        margin_sample=[0.008, 0.006, 0.006, 0.017],
        entropy_sample=[0.004, 0.011, 0.055, 0.056],
        random=[0.005, 0.004, 0.005, 0.005]
    )

    fig = make_subplots(rows=1, cols=1, subplot_titles=[''])

    al_names = ['random_sample', 'entropy_sample', 'margin_sample', 'bald', 'coreset', 'badge', 'PDAOPL']
    for idx, al_name in enumerate(al_names):
        al_traces = get_line_trace_with_err(query_size, acc[al_name], err[al_name], al_name, showlegend=True,
                                            scale=True)
        for trace in al_traces:
            fig.add_trace(trace, row=1, col=1)

    xaxis_layout = dict(
        title='Query Size',
        type='log',
        showgrid=True,
        showline=True,
        mirror=True,
        tickmode='array',
        ticks='outside',
        tickfont=dict(size=18),
        linecolor='rgba(0,0,0,1)',
        gridcolor='rgba(0,0,0,0.1)',
        showticklabels=True,
        titlefont=dict(size=20))

    yaxis_layout = dict(
        title='Test Accuracy',
        showgrid=True,
        showline=True,
        mirror=True,
        tickmode='array',
        ticks='outside',
        tickfont=dict(size=18),
        tickvals=[10 * i for i in range(11)],
        linecolor='rgba(0,0,0,1)',
        gridcolor='rgba(0,0,0,0.1)',
        showticklabels=True,
        titlefont=dict(size=20))

    fig['layout']['xaxis'].update(xaxis_layout)
    fig['layout']['yaxis'].update(yaxis_layout)

    fig.update_layout(
        autosize=False,
        width=450,
        height=350,
        legend=dict(
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=13),
            yanchor="bottom",
            y=0.06,
            xanchor="left",
            x=0.06,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0)',
        margin=dict(l=0, r=0, t=25, b=0),
        font=dict(family="Times")
    )
    fig.show()
    return fig

if __name__ == '__main__':
    print('start')
    if not os.path.exists('output'):
        os.makedirs('output')

    fig = robustness_to_query_emnist()
    fig.write_image('output/query_size.pdf')