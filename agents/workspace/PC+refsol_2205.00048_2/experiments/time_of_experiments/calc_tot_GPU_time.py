tot_time = 0

with open('Total_GPU_time.txt', 'r') as f:
    lines = f.readlines()

    for line in lines:
        line = line.split()
        val = float(line[-1])
        print(val)
        tot_time += val

tot_time_h = tot_time/60/60
print('Total running time for all experiments = ', str(tot_time_h), ' h')
        

