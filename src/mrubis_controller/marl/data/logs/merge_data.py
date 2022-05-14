import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from torch import ceil

dir_path = os.path.dirname(os.path.realpath(__file__))


def build_plot(data, title, path):
    colors = ['#B1063A', '#134293', '#058B79', '#DD9108', '#009e61',
              '#5a6065', '#00799e', '#f6ba00', '#b10639', '#dd6108']
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=16)
    # axes = plt.gca()
    # axes.set_ylim([0, 20])
    i = 0
    for label, d in data.items():
        x = np.array([i for i in range(len(d))])
        d[0] = d[0] if d[0] > 0 else 0
        for index, value in enumerate(d):
            if value < 0:
                d[index] = d[index - 1]
        plt.plot(x, [float(value) for value in d], colors[i % 10], label=label)
        i += 1
    plt.legend()
    plt.savefig(path)
    plt.clf()
    plt.close()


def build_data(input_data):
    data = {}
    for setup in input_data:
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
            data[setup] = plot
    return data


def func_no_c(x, a, b):
    return a * np.exp(-b * x) + 1


def func_c(x, a, b, c):
    return a * np.exp(-b * x) + c


def build_fitted_data(data, flexible_c):
    func = func_c if flexible_c else func_no_c
    new_data = {}
    parameters = {}
    for label, d in data.items():        
        x = np.array([i for i in range(len(d))])        
        d[0] = d[0] if d[0] > 0 else 0
        for index, value in enumerate(d):
            if value < 0:
                d[index] = d[index - 1]
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


def find_convergence(parameters, name, flexible_c):
    value_for_derivative = value_for_derivative_c if flexible_c else value_for_derivative_no_c
    c_path = '_flexible_c' if flexible_c else ''
    convergence_values = {}
    with open(f'{dir_path}/data_analysis/{name}/convergence{c_path}.txt', 'w') as f:
        for e in range(1, 6):
            epsilon = -pow(10, -e)
            convergence_values[epsilon] = {}
            f.write(f'Now testing epsilon={epsilon}\n')
            min_x = np.Inf
            min_label = ''
            for label, p in parameters.items():
                x = value_for_derivative(epsilon, *p)
                convergence_values[epsilon][label] = x
                if x < min_x:
                    min_x = x
                    min_label = label
                f.write(f'Setup {label} reaches the derivative at {x}\n')
            f.write(f'For epsilon={epsilon}, setup {min_label} converges fastest at {min_x}\n\n')
        f.close()
    return convergence_values


def find_outlier_values(data, convergence_values, name, flexible_c):
    c_path = '_flexible_c' if flexible_c else ''
    with open(f'{dir_path}/data_analysis/{name}/outliers{c_path}.txt', 'w') as f:
        for epsilon, c in convergence_values.items():
            for epsilon2 in range(0, 6):
                epsilon2 /= 10.
                min_frequency = np.Inf
                min_label = ''
                f.write(f'Now testing epsilon={epsilon}, epsilon2={epsilon2}\n')
                for label, d in data.items():
                    convergence_point = c[label]
                    relevant_values = d[int(np.ceil(convergence_point)):]
                    relevant_count = len(relevant_values)
                    outliers = [v for v in relevant_values if v > 1 + epsilon2]
                    outlier_count = len(outliers)
                    frequency = outlier_count / relevant_count
                    f.write(f'Outlier count for {label} post-convergence is {outlier_count} out of {relevant_count}, or a relative frequency of {frequency}.\n')
                    if frequency < min_frequency:
                        min_frequency = frequency
                        min_label = label
                f.write(f'Smallest outlier frequency for epsilon={epsilon}, epsilon2={epsilon2} is reached by {min_label} with {min_frequency}\n\n')
            f.write('\n')
        f.close()


def analyze(input_data, name, title, flexible_c=False):
    try:
        os.makedirs(f'{dir_path}/data_analysis/{name}/')
    except FileExistsError:
        print('Directory already exists.')
    data = build_data(input_data)
    path = f'{dir_path}/data_analysis/{name}/plot.png'
    c_path = '_flexible_c' if flexible_c else ''
    fitted_path = f'{dir_path}/data_analysis/{name}/plot_fitted{c_path}.png'
    fitted_data, parameters = build_fitted_data(data, flexible_c)
    build_plot(data, title, path)
    c_title = ' (flexible c)' if flexible_c else ''
    build_plot(fitted_data, f'{title} (fitted){c_title}', fitted_path)
    convergence_values = find_convergence(parameters, name, flexible_c)
    find_outlier_values(data, convergence_values, name, flexible_c)


flexible_c = False
analyze(["old_architecture", "new_architecture"], 'comparison_architecture',
        'Architecture comparison', flexible_c=flexible_c)
analyze(["1_agent_10_shops", "2_agents_5_shops", "3_agents_3.33_shops",
        "5_agents_2_shops", "10_agents_1_shop"],
        'comparison_10_shops', 'Setups for 10 shops', flexible_c=flexible_c)
analyze(["1_agent_15_shops", "2_agents_7.5_shops", "3_agents_5_shops",
        "5_agents_3_shops", "15_agents_1_shop"],
        'comparison_15_shops', 'Setups for 15 shops', flexible_c=flexible_c)
analyze(["1_agent_20_shops", "2_agents_10_shops", "4_agents_5_shops",
        "5_agents_4_shops", "10_agents_2_shops", "20_agents_1_shop"],
        'comparison_20_shops', 'Setups for 20 shops', flexible_c=flexible_c)
