import matplotlib.pyplot as plt
import os


allowed_ids = [1]

entry_dic: dict[float, dict[float, list[float]]] = {}

with os.scandir('.') as entries:
    for file in entries:
        if not file.name.endswith('.txt'):
            continue

        if len(file.name) < 18:
            print(f'What is this? {file.name}')
            continue

        # Only include env
        env_id = int(file.name.split('_')[3][5])
        if env_id not in allowed_ids:
            continue

        # alpha = float(file.name.split('_')[2].replace(',', '.'))

        # Path to saves
        with open(file.path) as f:
            lines = f.readlines()

        # Reconstruct values dictionary from source
        source = ''.join(lines)
        entries: list[tuple[dict[str, float], float]] = eval(source)

        for (params, cost) in entries:
            mass = params['body_mass']
            damp = params['dof_damping']

            if mass not in entry_dic:
                entry_dic[mass] = {}

            if damp not in entry_dic[mass]:
                entry_dic[mass][damp] = []

            entry_dic[mass][damp].append(cost)

    entries_avg: dict[float, dict[float, float]] = {}

    for mass, dd in entry_dic.items():
        entries_avg[mass] = {}
        for damp, cost_list in dd.items():
            avg_cost = sum(cost_list) / len(cost_list)
            entries_avg[mass][damp] = avg_cost

    masses_to_include = [0.5, 0.93, 1.21, 1.5]
    # markers = ['o', '^', '*', 's']
    markers = [' ', ' ', ' ', ' ']
    linestyles = ['-', '--', ':', '-.']
    i = 0
    for mass, dd in entries_avg.items():
        if min([abs(m - mass) for m in masses_to_include]) > 0.1:
            continue

        damps = entries_avg[mass].keys()
        costs = entries_avg[mass].values()
        plt.plot(
            damps, costs, marker=markers[i], linestyle=linestyles[i],
            label=f'Mass = {mass}')
        i += 1

    plt.legend()
    plt.xlabel('Friction')
    plt.ylabel('Cost')
    plt.savefig('pareto.pdf')

    # backwards = {}
    # for mass, dd in entries_avg.items():
    #     for damp, cost in dd.items():
    #         if damp not in backwards:
    #             backwards[damp] = {}

    #         backwards[damp][mass] = cost

    # for damp, dd in backwards.items():
    #     masses = backwards[damp].keys()
    #     costs = backwards[damp].values()
    #     plt.plot(masses, costs, label=f'Damp = {damp}')

    # plt.legend()
    # plt.xlabel('Mass')
    # plt.ylabel('Cost')
    # plt.show()
