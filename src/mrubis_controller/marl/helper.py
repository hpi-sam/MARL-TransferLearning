import datetime

import matplotlib.pyplot as plt
import numpy as np


def get_current_time():
    return "5_variance_clamp_9"
    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d_%H_%M')


def build_reward_plot(base_dir, reward_data, episode, shop_distribution):
    data = {}
    for agent in shop_distribution:
        for shop in agent:
            data[shop] = [d[0][shop] for d in reward_data]
    build_plot(data, f"Rewards till {episode}", f"{base_dir}/reward")
    write_data(len(reward_data), data, 'reward', base_dir)


def build_count_plot(base_dir, count_data, episode, shop_distribution):
    data = {}
    for index, agent in enumerate(shop_distribution):
        avgs = []
        for i in range(episode):
            filter_counts = list((filter(lambda x: x != -1, [count_data[shop][i] for shop in agent])))
            avg = -1 if len(filter_counts) == 0 else sum(filter_counts) / len(filter_counts)
            avgs.append(avg)
        data['Agent ' + str(index)] = avgs
    build_plot(data, f"Tries to find correct issue", f"{base_dir}/tries", "Number of Tries")
    write_data(len(list(count_data.values())[0]), data, 'tries', base_dir)


def build_loss_plot(base_dir, loss_data, episode):
    prepared_data = {}
    for loss in loss_data:
        for key, value in loss.items():
            if key not in prepared_data:
                prepared_data[key] = []
            if value:
                prepared_data[key].append(value[0])

    for agent in prepared_data.keys():
        title = f"Loss for agent #{agent} (episode {episode})"
        for network in ['actor', 'critic']:
            data = {
                network: [loss[network] for loss in prepared_data[agent]]
            }
            path = f"{base_dir}/{network}loss_agent_{agent}"
            build_plot(data, title, path)


def build_reward_plot(base_dir, reward_data, episode, shop_distribution):
    data = {}
    for agent in shop_distribution:
        for shop in agent:
            data[shop] = [d[0][shop] for d in reward_data]
    build_plot(data, f"Rewards till {episode}", f"{base_dir}/reward")
    write_data(len(reward_data), data, 'reward', base_dir)


def build_regret_plot(base_dir, regret_data, episode, shop_distribution, train_mode=True, avg_over_agents=False):
    data = {}
    if not avg_over_agents:
        for index, agent in enumerate(shop_distribution):
            avgs = []
            avg = 1
            for i in range(episode):
                counts = [regret_data[i][index][shop] for shop in regret_data[i][index]]
                if len(counts) != 0:
                    avg = sum(counts) / len(counts)
                avgs.append(avg)
            data['Agent ' + str(index)] = avgs
    else:
        avgs = []
        avg = 1
        for i in range(episode):
            counts = [regret_data[i][index][shop]
                      for index, agent in enumerate(shop_distribution) for shop in regret_data[i][index]]
            if len(counts) != 0:
                avg = sum(counts) / len(counts)
            avgs.append(avg)
        data['Shops '] = avgs

    title = "Regret per Episode for "
    path = f"{base_dir}/regret"
    # write_data(len(regret_data), regret_data, 'regret', base_dir)
    # print(data)
    with open(f"{path}.txt", "w+") as file:
        for i in data:
            file.write(str(i) + '\n')
        file.close()

    if not train_mode:
        with open(f"{path}_total_average.txt", "w+") as file:
            file.write(str(sum(data) / len(data)) + '\n')
            file.close()

    with open(f'{path}_average.txt', 'w+') as file:
        for i in range(episode):
            average = 0
            count = 0
            for regrets in regret_data[i].values():
                for regret in regrets.values():
                    average += regret
                    count += 1
            average /= count
            file.write(f'{str(average)}\n')
        file.close()

    colors = ['#B1063A', '#134293', '#058B79', '#DD9108', '#009e61',
              '#5a6065', '#00799e', '#f6ba00', '#b10639', '#dd6108']

    '''plt.figure(figsize=(8, 6))
    for agent, agent_data in data.items():
        plt.title(title + agent, fontsize=16)

        x = np.array([i for i in range(len(agent_data))])

        smoothed_wert = max(10, len(data))
        smoothed = np.convolve(agent_data, np.ones(smoothed_wert) / smoothed_wert, 'value')
        x_new = x[5:-4]
        plt.plot(x, agent_data, colors[9], label="exact regret")
        plt.plot(x_new, smoothed, colors[8], label="moving average")
        plt.xlabel('Episode')
        plt.ylabel('Regret')
        plt.legend()
        plt.savefig(path + agent)
        plt.clf()
    plt.close()'''


def build_plot(data, title, path, ylabel=None):
    '''colors = ['#B1063A', '#134293', '#058B79', '#DD9108', '#009e61',
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
        plt.plot(x, [float(value) for value in d], colors[9 - (i % 10)], label=label)
        i += 1
    plt.xlabel('Episode')
    if ylabel != None:
        plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(path)
    plt.clf()
    plt.close()'''


def write_data(count, data, title, base_dir):
    with open(f"{base_dir}/{title}.txt", "w+") as file:
        for i in range(count):
            data_string = ''.join((str(data[shop][i]) if data[shop][i] > 0 else '') + ',' for shop in data)
            file.write(data_string + '\n')


def build_plot_from_file(titles, path, output_path, base_dir='./mrubis_controller/marl/data/logs'):
    data = {}
    with open(f'{base_dir}/{path}.txt') as file:
        lines = file.readlines()
        for index, title in enumerate(titles, start=1):
            data[title] = [line.split(',')[index] for line in lines]
    build_plot(data, 'Number of tries', f'{base_dir}/{output_path}.png')


def print_logs(base_dir, logs):
    path = f"{base_dir}/actions.txt"
    with open(path, 'w+') as file:
        for t in range(len(logs)):
            t_data = logs[t]
            for inner_t in range(len(t_data)):
                data = t_data[inner_t]
                file.write(f'Episode: {t}; Try: {inner_t};\n')
                for shop in data:
                    file.write(f'\t{shop}: {data[shop]["Root causes"]} / {data[shop]["Actions"]}: {data[shop]["Reward"]}\n')
        file.close()


if __name__ == "__main__":
    build_plot_from_file(['mrubis #1'], '2022_02_04_13_33/tries', 'robustness_shop')
