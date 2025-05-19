

n_elements = 10 * 10
visit_each_n_steps = 31
counter_of_visits = [0] * n_elements
i = 0
for _ in range(2):
    for j in range(n_elements):
        if i % visit_each_n_steps == 0:
            counter_of_visits[j] += 1
            print(j)
        i += 1


print(counter_of_visits)
