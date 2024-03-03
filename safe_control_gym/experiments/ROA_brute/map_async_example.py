
from multiprocessing import Pool, Queue, cpu_count
import time

print("Number of cpu : ", cpu_count())

def func(x, result_q):
    value = x*x
    result = {"x": x, "result": value}
    result_q.put(result)

    return x*x


def main():
    # create results queue
    results_q = Queue()
    results = []
    # create a pool of workers
    pool = Pool(processes=4)
    inputs = range(10)
    print('inputs type', type(inputs))
    print(inputs)
    r = pool.map_async(func, inputs)
    # DO STUFF
    # print('HERE')
    # print('MORE')
    print(r)
    # print('DONE')

if __name__ == '__main__':
    main()