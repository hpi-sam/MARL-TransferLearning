from cmath import nan
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from functools import reduce

from sklearn.metrics import jaccard_score

dir_path = os.path.dirname(os.path.realpath(__file__))


def build_plot(data, title, path, x=np.array([i for i in range(400)]), ylabel=None):
    colors = ['#B1063A', '#134293', '#058B79', '#DD9108', '#009e61',
              '#5a6065', '#00799e', '#f6ba00', '#b10639', '#dd6108']
    plt.figure(figsize=(8, 6))
    #plt.title(title, fontsize=16)
    i = 0
    for label, d in data.items():
        y = [float(value) for value in d]
        plt.plot(x, y, colors[9 - (i % 10)], label=label)
        i += 1
    plt.xlabel('Episode', fontsize=16)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    plt.savefig(path)
    plt.clf()
    plt.close()


def get_print_name(name):
    print_name = name.replace('_', ' ').replace('agents ', 'agents, ').replace('agent ', 'agent, ').replace('mod', '(modulus)')
    if name[-2:] == '_r':
        print_name = print_name[:-2]
    return print_name


def build_data(input_data):
    data = {}
    regret_data = {}
    for setup in input_data:            
        print_setup = get_print_name(setup)
        with open(f'{dir_path}/{setup}/tries.txt') as f:
            lines = f.read().splitlines()
            plot = []
            for line in lines:
                values = line.split(',')
                numbers = []
                for value in values:
                    if value != '':
                        numbers.append(float(value))
                plot.append(sum(numbers)/len(numbers))                        
            plot[0] = plot[0] if plot[0] > 0 else 0
            for index, value in enumerate(plot):
                if value < 0:
                    plot[index] = plot[index - 1]
            data[print_setup] = plot
            f.close()
        with open(f'{dir_path}/{setup}/regret_average.txt') as f:
            lines = f.read().splitlines()
            plot = []
            for line in lines:
                plot.append(float(line))
            regret_data[print_setup] = plot
            f.close()
    return data, regret_data


def func_no_c(x, a, b):
    return a * np.exp(-b * x) + 1


def func_c(x, a, b, c):
    return a * np.exp(-b * x) + c


def sigmoid(x, a, b, k, m, v):
    return a + (k - a)/((1 + np.exp(-b * (x - m))) ** (1/v))


def build_fitted_data(data, flexible_c):
    func = func_c if flexible_c else func_no_c
    return fit(data, func)


def build_fitted_regret(data, episodes):
    x = np.array([i for i in range(episodes)])
    new_data = {}
    parameters = {}
    for label, d in data.items():
        try:
            popt, pcov = curve_fit(sigmoid, x, d,
                                   bounds=([0, -np.Inf, 0, -100, 0], [np.inf, np.inf, np.inf, np.inf, np.Inf]),
                                   p0 = [1, 0.5, 0, 100, 1])
            # we assume b >= 0. A b < 0 effect can be gained by switching a and k
            # we assume a, k >= 0 since regret should be nonnegative
            # some setups that perform very poorly land at k = 0 and use this limit, but given that these setups don't have significant descent to begin with this is not critical
            # we assume m >= -100 as a limiter, though this value can be changed
            # for setups that perform very well, this limit is exhausted and could be decreased further, but it will not have a huge effect on the actual curve
            # Initial values are set such that a > k (so we have a decreasing sigmoid), and with both (natural) extremes of the regret spectrum for a and k
            new_data[label] = sigmoid(x, *popt)
            parameters[label] = popt
        except RuntimeError as error:
            print(f'Label {label} does not converge: {error} Trying exponential.')
            popt, pcov = curve_fit(func_c, x, d)
            new_data[label] = func_c(x, *popt)
            parameters[label] = popt
    return new_data, parameters


def fit(data, func):
    new_data = {}
    parameters = {}
    for label, d in data.items():
        x = np.array([i for i in range(len(d))])
        popt, pcov = curve_fit(func, x, d)
        new_data[label] = func(x, *popt)
        parameters[label] = popt
    return new_data, parameters

# d/dx a * np.exp(-b * x) + 1 = -a * b * np.exp(-b * x)
# -a * b * np.exp(-b * x) = d
# np.exp(-b * x) = -d / (a * b)
# -b * x = np.log(-d / (a * b))
# x = np.log(-a * b / d) / b


def value_for_derivative_no_c(d, a, b):
    return np.log(-a * b / d) / b


def value_for_derivative_c(d, a, b, c):
    return np.log(-a * b / d) / b


def value_for_sigmoid(y, a, b, k, m, v):
    if y >= max(a, k) or y <= min(a, k):
        return nan
    return -np.log(((a - k)/(a - y)) ** v - 1)/b + m


def find_convergence(parameters, name, title, flexible_c):
    value_for_derivative = value_for_derivative_c if flexible_c else value_for_derivative_no_c
    c_path = '_flexible_c' if flexible_c else ''
    convergence_values = {}
    with open(f'{dir_path}/data_analysis/{name}/convergence{c_path}.txt', 'w') as f:
        with open(f'{dir_path}/data_analysis/{name}/convergence_latex{c_path}.txt', 'w') as latex_file:            
            latex_file.write('\\begin{table}[h]\n\\begin{small}\n')
            latex_file.write('\\begin{tabular}{l' + ('c' * 5) + '}\\toprule\n')
            latex_file.write('    & ' + reduce(lambda a, b: f'{a} & {b}', ['$-10^{-' + str(i) + '}$' for i in np.arange(1, 6)]) + ' \\\\ \\midrule\n')
            label_lines = {}
            for e in range(1, 6):
                epsilon = -pow(10, -e)
                convergence_values[epsilon] = {}
                f.write(f'Now testing epsilon={epsilon}\n')
                min_x = np.Inf
                min_label = ''
                for label, p in parameters.items():
                    if e == 1:
                        label_lines[label] = f'    {label}'
                    x = value_for_derivative(epsilon, *p)
                    label_lines[label] += f' & {x:.3f}'
                    convergence_values[epsilon][label] = x
                    if x < min_x:
                        min_x = x
                        min_label = label
                    f.write(f'Setup {label} reaches the derivative at {x}\n')
                f.write(f'For epsilon={epsilon}, setup {min_label} converges fastest at {min_x}\n\n')
            latex_file.write(reduce(lambda a, b: f'{a} \\\\\n{b}', label_lines.values()))
            latex_file.write(' \\\\\n    \\bottomrule\n\\end{tabular}\n\\centering\n\\end{small}\n\\caption{Convergence points for ' + title + '. The earliest convergence for each $\\varepsilon$ is bolded.}\n\\end{table}') 
            f.close()
            latex_file.close()
    return convergence_values


def find_outlier_values(data, convergence_values, name, title, flexible_c):
    c_path = '_flexible_c' if flexible_c else ''
    with open(f'{dir_path}/data_analysis/{name}/outliers{c_path}.txt', 'w') as f:
        with open(f'{dir_path}/data_analysis/{name}/outliers_latex{c_path}.txt', 'w') as latex_file:
            latex_file.write('\\begin{table}[h]\n\\begin{small}\n')
            latex_file.write('\\begin{tabular}{l' + ('c' * 5) + '}\\toprule\n')
            # latex_file.write('    & ' + reduce(lambda a, b: f'{a} & ' + np.format_float_positional(b, trim='-'), np.arange(0, 6) / 10.) + ' \\\\ \\midrule\n')
            latex_file.write('    & ' + reduce(lambda a, b: f'{a} & {b}', ['$-10^{-' + str(i) + '}$' for i in np.arange(1, 6)]) + ' \\\\ \\midrule\n')
            j = 1
            latex_lines = {}
            for epsilon, c in convergence_values.items():
                # latex_file.write('    $-10^{-' + str(j) + '}$')
                for epsilon2 in range(0, 6):
                    epsilon2 /= 10.
                    if epsilon == -0.1:
                        latex_lines[str(epsilon2)] = f'    {epsilon2}'
                    min_frequency = np.Inf
                    min_label = ''
                    min_id = 0
                    f.write(f'Now testing epsilon={epsilon}, epsilon2={epsilon2}\n')
                    i = 0
                    for label, d in data.items():
                        convergence_point = max(c[label], 0)
                        relevant_values = d[int(np.ceil(convergence_point)):]
                        relevant_count = len(relevant_values)
                        outliers = [v for v in relevant_values if v > 1 + epsilon2]
                        outlier_count = len(outliers)
                        if relevant_count == 0:
                            f.write(f'Outlier count for {label} cannot be determined, as there are no relevant values.\n')
                        else:
                            frequency = outlier_count / relevant_count
                            f.write(f'Outlier count for {label} post-convergence is {outlier_count} out of {relevant_count}, or a relative frequency of {frequency}.\n')
                            if frequency < min_frequency:
                                min_frequency = frequency
                                min_label = label
                                min_id = i
                        i += 1
                    f.write(f'Smallest outlier frequency for epsilon={epsilon}, epsilon2={epsilon2} is reached by {min_label} with {min_frequency}\n\n')
                    if min_frequency != np.inf:
                        latex_lines[str(epsilon2)] += f' & {min_frequency:.3f} (\#{min_id})'
                    else:                        
                        latex_lines[str(epsilon2)] += f' & ---'
                f.write('\n')
                j += 1
            f.close()
            latex_file.write(reduce(lambda a, b: f'{a} \\\\\n{b}', latex_lines.values()))
            latex_file.write(' \\\\ \n    \\bottomrule\n\\end{tabular}\n\\centering\n\\end{small}\n\\caption{Fewest relative outliers for ' + title + '}\n\\end{table}')
            latex_file.close()


def smooth_regret_data(regret_data):
    smoothed_data = {}
    for label, r in regret_data.items():
        smoothed_wert = 10
        smoothed = np.convolve(r, np.ones(smoothed_wert) / smoothed_wert, 'value')
        smoothed_data[label] = smoothed
    return smoothed_data


def fit_tries(data, name, title, flexible_c):    
    c_path = '_flexible_c' if flexible_c else ''
    fitted_path = f'{dir_path}/data_analysis/{name}/plot_tries_fitted{c_path}.png'
    fitted_data, parameters = build_fitted_data(data, flexible_c)
    c_title = ' (Flexible c)' if flexible_c else ''
    build_plot(fitted_data, f'{title} (Fitted){c_title}', fitted_path, ylabel='Tries')
    convergence_values = find_convergence(parameters, name, title, flexible_c)
    find_outlier_values(data, convergence_values, name, title, flexible_c)


def analyze_regret_data(regret_parameters, name, title, episodes):
    with open(f'{dir_path}/data_analysis/{name}/regret_analysis.txt', 'w') as f:
        with open(f'{dir_path}/data_analysis/{name}/regret_analysis_latex.txt', 'w') as latex_file:
            latex_file.write('\\begin{table}[h]\n\\begin{small}\n')
            latex_file.write('\\begin{tabular}{l' + ('c' * 6) + '}\\toprule\n')
            latex_file.write('    & ' + reduce(lambda a, b: f'{a} & ' + np.format_float_positional(b, trim='-'), np.arange(4, 9) / 10.) + ' \\\\ \\midrule\n')
            best_values = np.full(10, np.Inf)
            best_labels = [None] * 10
            best_lower_asymptote = np.Inf
            best_lower_asymptote_label = ''
            for label, p in regret_parameters.items():
                latex_file.write(f'    {label} ')
                f.write(f'Analysis for {label}\n')
                a, b, k, m, v = p
                f.write(f'a={a}, k={k}, b={b}, m={m}, v={v}\n')
                f.write(f'Turning point at m={m}\n')
                if m <= 0.75 * episodes:
                    f.write(f'Lower asymptote is at {min(a, k)}\n')
                    if min(a, k) <= best_lower_asymptote:
                        best_lower_asymptote = min(a, k)
                        best_lower_asymptote_label = label
                else:
                    f.write('Cannot determine lower asymptote from data.\n')
                for i in range(4, 9):
                    y = i / 10
                    x = value_for_sigmoid(y, *p)
                    if x is nan or x > episodes:
                        f.write(f'y={y}: Sigmoid for {label} never reaches it.\n')
                        latex_file.write('& --- ')
                    else:
                        f.write(f'y={y}: Sigmoid for {label} reaches regret at x={x}\n')
                        latex_file.write(f'& {x:.2f} ')
                        if x <= best_values[i]:
                            best_values[i] = x
                            best_labels[i] = label
                f.write('\n')
                latex_file.write('\\\\\n')
            f.write(f'Best lower asymptote: {best_lower_asymptote_label} with {best_lower_asymptote}\n')
            for i in range(10):
                if best_values[i] != np.Inf:
                    f.write(f'Best setup for y={i / 10}: {best_labels[i]} at episode {best_values[i]}\n')
            f.close()
            latex_file.write('    \\bottomrule\n\\end{tabular}\n\\centering\n\\end{small}\n\\caption{Regret analysis for ' + title + '. The earliest setup to reach the given threshold is bolded.}\n\\end{table}')
            latex_file.close()


def analyze_data(data, regret_data, name, title, episodes):
    path = f'{dir_path}/data_analysis/{name}/plot_tries_raw.png'
    build_plot(data, f'{title} (Tries)', path, ylabel='Tries', x=np.array([i for i in range(episodes)]))
    fit_tries(data, name, title, True)
    fit_tries(data, name, title, False)
    smoothed_regret = smooth_regret_data(regret_data)
    path_regret = f'{dir_path}/data_analysis/{name}/plot_regret.png'
    path_regret_raw = f'{dir_path}/data_analysis/{name}/plot_regret_raw.png'
    build_plot(regret_data, f'{title} (Raw regret)', path_regret_raw, ylabel='Regret', x=np.array([i for i in range(episodes)]))
    build_plot(smoothed_regret, f'{title} (Moving average regret)', path_regret, ylabel='Regret', x=np.array([i for i in range(episodes)])[5:-4])
    fitted_regret, regret_parameters = build_fitted_regret(regret_data, episodes)
    fitted_regret_path = f'{dir_path}/data_analysis/{name}/plot_regret_fitted.png'
    build_plot(fitted_regret, f'{title} (Fitted regret)', fitted_regret_path, ylabel='Regret', x=np.array([i for i in range(episodes)]))
    analyze_regret_data(regret_parameters, name, title, episodes)


def analyze(input_data, name, title, episodes):
    try:
        os.makedirs(f'{dir_path}/data_analysis/{name}/')
    except FileExistsError:
        print('Directory already exists.')
    data, regret_data = build_data(input_data)
    analyze_data(data, regret_data, name, title, episodes)


def analyze_average(input_data, name, title, episodes, samples=10):    
    try:
        os.makedirs(f'{dir_path}/data_analysis/{name}/')
    except FileExistsError:
        print('Directory already exists.')
    data = {}
    regret_data = {}
    for setup in input_data:
        sub_setups = [f'{setup}_{i}' for i in range(samples)]
        analyze(sub_setups, f'{setup}_raw', 'Samples for reward variance',
                episodes)
        sub_data, sub_regret_data = build_data(sub_setups)
        sub_data_matrix = np.array(
            [sub_data[s] for s in sub_data.keys()])
        sub_regret_data_matrix = np.array(
            [sub_regret_data[s] for s in sub_data.keys()])
        setup_data = np.average(sub_data_matrix, axis=0)
        setup_regret_data = np.average(sub_regret_data_matrix, axis=0)
        data[get_print_name(setup)] = setup_data
        regret_data[get_print_name(setup)] = setup_regret_data
    analyze_data(data, regret_data, name, title, episodes)


episodes = 400
samples = 10
analyze(["old_architecture_r", "new_architecture_r", "new_architecture_mod_r"],
         'comparison_architecture_regret', 'Architecture comparison', episodes)
analyze(["1_agent_10_shops_r", "2_agents_5_shops_r", "3_agents_3.33_shops_r",
        "5_agents_2_shops_r", "10_agents_1_shop_r"],
        'comparison_10_shops_regret', 'Setups for 10 shops', episodes)
analyze(["1_agent_15_shops_r", "2_agents_7.5_shops_r", "3_agents_5_shops_r",
        "5_agents_3_shops_r", "15_agents_1_shop_r"],
        'comparison_15_shops_regret', 'Setups for 15 shops', episodes)
analyze(["1_agent_20_shops_r", "2_agents_10_shops_r", "4_agents_5_shops_r",
        "5_agents_4_shops_r", "10_agents_2_shops_r", "20_agents_1_shop_r"],
        'comparison_20_shops_regret', 'Setups for 20 shops', episodes)
analyze(["1_agent_40_shops_r", "2_agents_20_shops_r", "4_agents_10_shops_r",
         "5_agents_8_shops_r", "8_agents_5_shops_r", "10_agents_4_shops_r",
         "20_agents_2_shops_r", "40_agents_1_shop_r"],
        'comparison_40_shops_regret', 'Setups for 40 shops', episodes)
analyze(["1_agent_80_shops_r", "2_agents_40_shops_r", "4_agents_20_shops_r",
         "8_agents_10_shops_r", "16_agents_5_shops_r",
         "20_agents_4_shops_r", "40_agents_2_shops_r", "80_agents_1_shop_r"],
        'comparison_80_shops_regret', 'Setups for 80 shops', episodes)
analyze(["1_agent_10_shops_r", "1_agent_15_shops_r", "1_agent_20_shops_r",
         "1_agent_40_shops_r", "1_agent_80_shops_r"],
        'comparison_1_agent_regret', 'Setups for 1 agent', episodes)
analyze(["2_agents_5_shops_r", "2_agents_7.5_shops_r", "2_agents_10_shops_r",
         "2_agents_20_shops_r", "2_agents_40_shops_r"],
        'comparison_2_agents_regret', 'Setups for 2 agents', episodes)
analyze(["4_agents_5_shops_r", "4_agents_10_shops_r", "4_agents_20_shops_r"],
        'comparison_4_agents_regret', 'Setups for 4 agents', episodes)
analyze(["10_agents_1_shop_r", "15_agents_1_shop_r", "20_agents_1_shop_r",
         "40_agents_1_shop_r", "80_agents_1_shop_r"],
        'comparison_1_shop_per_agent_regret', 'Setups for 1 shop per agent',
        episodes)
analyze(["5_agents_2_shops_r", "10_agents_2_shops_r", "20_agents_2_shops_r",
         "40_agents_2_shops_r"],
        'comparison_2_shops_per_agent_regret', 'Setups for 2 shops per agent',
        episodes)
analyze(["5_agents_4_shops_r", "10_agents_4_shops_r", "20_agents_4_shops_r"],
        'comparison_4_shops_per_agent_regret', 'Setups for 4 shops per agent',
        episodes)
'''analyze(["0_variance", "1_variance", "2_variance", "3_variance",
        "5_variance", "10_variance"],
        'comparison_variance', 'Setups for different reward variance',
        episodes)'''
'''analyze(["0_variance_1", "1_variance_1"],
        'comparison_variance_help', 'Setups for different reward variance',
        episodes)'''
analyze_average(["0_variance", "1_variance", "2_variance", "3_variance",
                "5_variance", "5_variance_clamp"],
                'comparison_variance_average',
                'Setups for different reward variance; average of 10 samples',
                episodes, samples)
analyze_average(["5_variance", "5_variance_clamp"],
                'comparison_clamp_average',
                'Setups for different reward variance; average of 10 samples',
                episodes, samples)
