from collections import defaultdict

from plotly.subplots import make_subplots
import plotly
import numpy as np
import pandas as pd
import os
import sys


COLORS = ['rgb(31, 119, 180)',
          'rgb(255, 127, 14)',
          'rgb(44, 160, 44)',
          'rgb(214, 39, 40)',
          'rgb(148, 103, 189)',
          'rgb(140, 86, 75)',
          'rgb(227, 119, 194)',
          'rgb(127, 127, 127)',
          'rgb(188, 189, 34)',
          'rgb(23, 190, 207)',
          'rgb(230, 25, 75)',
          'rgb(60, 180, 75)',
          'rgb(255, 225, 25)',
          'rgb(67, 99, 216)',
          'rgb(245, 130, 49)',
          'rgb(145, 30, 180)',
          'rgb(70, 240, 240)',
          ]


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
    for data in data_list:
        mean = np.nanmean(data)
        if np.isnan(mean):
            continue
        y.append(mean)
        std = np.nanstd(data)
        y_err.append(std)
        high.append(mean + std)
        low.append(mean - std)

    low = low[::-1]
    #     print(high,low)
    return y, y_err, high, low


def get_res_seed(exp_dir):
    df = pd.read_csv('{}/{}'.format(exp_dir, 'train_result.csv'))
    col_acc = [name for name in df.columns if 'cc' in name and 'topk' not in name]
    acc_dict = df[col_acc].iloc[-1, :].to_dict()
    return acc_dict


def get_res_seeds(exp_dir):
    res = defaultdict(list)
    for seed in os.listdir(exp_dir):
        res_seed = get_res_seed('{}/{}'.format(exp_dir, seed))
        for k, v in res_seed.items():
            res[k].append(v)
    return res


def get_res_method(root_dir):
    model_sizes = [1, 2, 4, 6, 8, 10, 12]
    res = defaultdict(list)
    for size in model_sizes:
        exp_dir = '{}/{}'.format(root_dir, 'wrn_D16_W{}'.format(size))
        res_model = get_res_seeds(exp_dir)
        for k, v in res_model.items():
            res[k].append(v)
    return res


def get_line_trace(x, y_list, idx, name, showlegend=True):
    color = COLORS[idx]

    y, y_err, high, low = get_data_uncertainty(y_list)
    x = x[:len(y)]
    x_rev = x[::-1]

    #     x_rev = x[::-1]
    #     y, y_err, high, low = get_data_uncertainty(y_list)

    traces = [
        dict(
            x=x,
            y=y,
            name=name,
            type='scatter',
            marker=dict(color=color, size=8),
            showlegend=showlegend,
            text=['{0:.3f}±{1:.3f}'.format(a, b)
                  for a, b in zip(y, y_err)],
        ),
        dict(
            x=x + x_rev,
            y=high + low,
            line=dict(color='rgba(255,255,255,0)'),
            name=name + '_fill',
            #                 legendgroup = config['name'],
            hoverinfo='skip',
            fill='toself',
            fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.2)'),
            showlegend=False,
        ),
    ]

    return traces


def get_res_wide(root_dir, attack_method='PGD'):
    res = get_res_method(root_dir)
    x_val = [0.0]
    for name in res.keys():
        if 'acc' in name and attack_method in name:
            x_val.append(float(name.split('-')[-1]))

    x_val.sort()
    widen_factors = [1, 2, 4, 6, 8, 10, 12]
    plot_res = {}

    for x in x_val:
        if x == 0.0:
            name = 'Valid Acc.'
        else:
            name = '{}_rtest_acc-{}'.format(attack_method, x)
        plot_res['ε={}'.format(x)] = res[name]

    return plot_res, widen_factors


def plot_res_wide(root_dir, attack_method='PGD'):
    fig = make_subplots(rows=1, cols=1, subplot_titles=[attack_method])
    res, x_val = get_res_wide(root_dir, attack_method)
    idx = -0
    for k, v in res.items():
        traces = get_line_trace(x_val, res[k], idx, name=k, showlegend=True)
        for trace in traces:
            fig.add_trace(trace, row=1, col=1)
        idx += 1

    xaxis_layout = dict(
        title='Widen Factor',
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
        #         tickvals=[10 * i for i in range(11)],
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
            yanchor="top",
            y=0.96,
            xanchor="left",
            x=1,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0)',
        margin=dict(l=0, r=0, t=25, b=0),
        font=dict(family="Times")
    )

    # fig.show()
    return fig


def get_line_sensitivity(x, y_list, idx, name, showlegend=True):
    color = COLORS[idx]

    y, y_err, high, low = get_data_uncertainty(y_list)
    x = x[:len(y)]
    x_rev = x[::-1]

    traces = [
        dict(
            x=x,
            y=y,
            name=name,
            type='scatter',
            marker=dict(color=color, size=8),
            showlegend=showlegend,
            text=['{0:.3f}±{1:.3f}'.format(a, b)
                  for a, b in zip(y, y_err)],
        ),
        dict(
            x=x + x_rev,
            y=high + low,
            line=dict(color='rgba(255,255,255,0)'),
            name=name + '_fill',
            hoverinfo='skip',
            fill='toself',
            fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.2)'),
            showlegend=False,
        ),
    ]

    return traces


def get_res_sens(root_dir, attack_method='PGD'):
    res = get_res_method(root_dir)
    x_val = [0]
    for name in res.keys():
        if 'acc' in name and attack_method in name:
            x_val.append(float(name.split('-')[-1]))

    x_val.sort()

    widen_factors = [1, 2, 4, 6, 8, 10, 12]
    plot_res = {'widen={}'.format(widen): [] for widen in widen_factors}

    for idx, widen in enumerate(widen_factors):
        plot_res['widen={}'.format(widen)].append(res['Valid Acc.'][idx])
        for x in x_val:
            if x == 0:
                continue
            name = '{}_rtest_acc-{}'.format(attack_method, x)
            plot_res['widen={}'.format(widen)].append(res[name][idx])

    return plot_res, x_val


def plot_sensitivity(root_dir, attack_method='PGD'):
    fig = make_subplots(rows=1, cols=1, subplot_titles=[attack_method])
    res, x_val = get_res_sens(root_dir, attack_method)
    idx = 0
    for k, v in res.items():
        traces = get_line_trace(x_val, res[k], idx, name=k, showlegend=True)
        for trace in traces:
            fig.add_trace(trace, row=1, col=1)
        idx += 1

    xaxis_layout = dict(
        title='Attack Strength',
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
        #         tickvals=[10 * i for i in range(11)],
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
            yanchor="top",
            y=0.96,
            xanchor="right",
            x=0.96,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0)',
        margin=dict(l=0, r=0, t=25, b=0),
        font=dict(family="Times")
    )

    # fig.show()
    return fig


if __name__ == '__main__':
    print('start!')
    if not os.path.exists('fig'):
        os.makedirs('fig')

    print('cifar10!')
    sys.stdout.flush()
    root_dir = '/h/zycluke/Project/adversarial_attack/checkpoint/exp_4_18_cifar10/cifar10/'
    fig = plot_res_wide(root_dir, attack_method='PGD')
    fig.write_image('fig/cifar10-pgd-width.pdf')
    fig = plot_res_wide(root_dir, attack_method='FGSM')
    fig.write_image('fig/cifar10-fgsm-width.pdf')
    fig = plot_res_wide(root_dir, attack_method='GN')
    fig.write_image('fig/cifar10-gn-width.pdf')

    fig = plot_sensitivity(root_dir, attack_method='PGD')
    fig.write_image('fig/cifar10-pgd-sens.pdf')
    fig = plot_sensitivity(root_dir, attack_method='FGSM')
    fig.write_image('fig/cifar10-fgsm-sens.pdf')
    fig = plot_sensitivity(root_dir, attack_method='GN')
    fig.write_image('fig/cifar10-gn-sens.pdf')

    print('fmnist!')
    sys.stdout.flush()
    root_dir = '/h/zycluke/Project/adversarial_attack/checkpoint/exp_4_18_fashion/fashionmnist/'
    fig = plot_res_wide(root_dir, attack_method='PGD')
    fig.write_image('fig/fmnist-pgd-width.pdf')
    fig = plot_res_wide(root_dir, attack_method='FGSM')
    fig.write_image('fig/fmnist-fgsm-width.pdf')
    fig = plot_res_wide(root_dir, attack_method='GN')
    fig.write_image('fig/fmnist-gn-width.pdf')

    fig = plot_sensitivity(root_dir, attack_method='PGD')
    fig.write_image('fig/fmnist-pgd-sens.pdf')
    fig = plot_sensitivity(root_dir, attack_method='FGSM')
    fig.write_image('fig/fmnist-fgsm-sens.pdf')
    fig = plot_sensitivity(root_dir, attack_method='GN')
    fig.write_image('fig/fmnist-gn-sens.pdf')
    #
    print('emnist!')
    sys.stdout.flush()
    root_dir = '/h/zycluke/Project/adversarial_attack/checkpoint/exp_4_18_emnist/emnist/'
    fig = plot_res_wide(root_dir, attack_method='PGD')
    # plotly.offline.plot(fig, filename='emnist-pgd-width.html', auto_open=False)
    fig.write_image('fig/emnist-pgd-width.pdf')
    fig = plot_res_wide(root_dir, attack_method='FGSM')
    # plotly.offline.plot(fig, filename='emnist-fgsm-width.html', auto_open=False)
    fig.write_image('fig/emnist-fgsm-width.pdf')
    fig = plot_res_wide(root_dir, attack_method='GN')
    # plotly.offline.plot(fig, filename='emnist-gn-width.html', auto_open=False)
    fig.write_image('fig/emnist-gn-width.pdf')

    fig = plot_sensitivity(root_dir, attack_method='PGD')
    # plotly.offline.plot(fig, filename='emnist-pgd-sens.html', auto_open=False)
    fig.write_image('fig/emnist-pgd-sens.pdf')
    fig = plot_sensitivity(root_dir, attack_method='FGSM')
    # plotly.offline.plot(fig, filename='emnist-fgsm-sens.html', auto_open=False)
    fig.write_image('fig/emnist-fgsm-sens.pdf')
    fig = plot_sensitivity(root_dir, attack_method='GN')
    # plotly.offline.plot(fig, filename='emnist-gn-sens.html', auto_open=False)
    fig.write_image('fig/emnist-gn-sens.pdf')

    print('finish!')
    sys.stdout.flush()
