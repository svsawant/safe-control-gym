import os
import time
import multiprocessing
import json

def task(args):
    count = args[0]
    queue = args[1]
    value = count ** 2
    # if the calculation is done, add to queue
    result = {'number': count, 'value': value} 
    print('adding {} number to queue'.format(count))
    queue.put(result)
    time.sleep(2)
    return value


def main():
    # init result queue
    manager = multiprocessing.Manager()
    q = manager.Queue()
    # init pool
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count - 2)

    # create a list of desired number
    number_list = [i for i in range(4)]

    # if the desired json file does not exist, create one
    if not os.path.exists('results.json'):
        # write a empty list to json file
        result = []
        with open('results.json', 'w') as f:
            json.dump(result, f, indent=4)
    elif os.path.exists('results.json'):
        '''
        if the desired json file exists,
        (possibly because the program is 
        terminated before all tasks are done)
        '''
        # read the result from json file
        with open('results.json', 'r') as f:
            result = json.load(f)
        # extract the existing number list
        exist_number_list = [x['number'] for x in result]
        # subtract the existing number list from the desired number list
        search_list = list(set(number_list) - set(exist_number_list))
        print('search_list', search_list)
    # send task to pool
    roa = pool.map_async(task, [(x, q) for x in search_list])
    time.sleep(5)
    all_done = False
    while not all_done:

        # check if all tasks are done
        with open('results.json', 'r') as f:
            result = json.load(f)
            
        # Grab all finished results from the queue
        while q.qsize() > 0:
            result.append(q.get())

        # save results to disk
        with open("results.json", "w") as f:
            json.dump(result, f, indent=4)
            
        # # manually terminate the pool
        # if len(result) == 2:
        #     exit()
        
        if len(result) == len(number_list):
            all_done = True
        
        
    # print('result', result)
    print('roa', roa.get())

    # sort_result = False
    sort_result = True

    if sort_result:
        # read the result from json file
        with open('results.json', 'r') as f:
            result = json.load(f)
        print('result before sorting', result)
        # sort the result list
        result.sort(key=lambda x: x['number'])
        print('result after sorting', result)
        # save the sorted result to json file
        with open('results.json', 'w') as f:
            json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
    