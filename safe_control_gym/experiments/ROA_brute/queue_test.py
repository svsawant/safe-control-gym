from multiprocessing import Queue, Pool
import json

colors = ['red', 'green', 'blue', 'black']
cnt = 1
# instantiating a queue object
queue = Queue()
print('pushing items to queue:')
for color in colors:
    print('item no: ', cnt, ' ', color)
    # put will push the item to the end of the queue
    queue.put(color)
    cnt += 1
    
x_cand = [1, 2, 3, 4]

def roi(x, result_q):
    # create env here
    stable = x**2 - 1
    result = {"x": x, "result": stable}
    result_q.put(result)

    return stable

def main():
    result_q = Queue()
    result = []
    p = Pool(4)
    p.map_async(roi, range(4))

    print('\npopping items from queue:')
    cnt = 0
    while not all_done:
        while not queue.empty():
            # get will pop the first item in the queue
            print('item no: ', cnt, ' ', queue.get()) 
            result = queue.get()
            cnt += 1
        with open('queue_test.json', 'w') as f:
            json.dump(result, f)

if __name__ == "__main__":
    main()