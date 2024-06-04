import multiprocessing, time

def task(args):
    count = args[0]
    queue = args[1]
    for i in range(count):
        print('adding {} number from count {}'.format(i, count))
        queue.put("{} number from count {}".format(i, count))
    return "Done"


def main():
    manager = multiprocessing.Manager()
    q = manager.Queue()
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count - 2)
    result = pool.map_async(task, [(x, q) for x in range(10)])
    time.sleep(1)
    while not q.empty():
        print(q.get())
    print(result.get()) # get all the result

if __name__ == "__main__":
    main()