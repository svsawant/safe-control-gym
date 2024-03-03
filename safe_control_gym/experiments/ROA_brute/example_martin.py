from multiprocessing import Pool, Queue
import json

def roi(x, y, result_queue):
    # create env here
    stable = env.roi(x, y)
    result = {"x": x, "y": y, "result": stable}
    result_queue.put(result)

    return x**2 + y**2 - 1


def main():
    results_queue = Queue()
    results = []
    p = Pool(4)
    p.map(roi, range(4))
    results_queue.save()
    p.map_async(roi, range(4))
    while not all_done:
        # Grab all finished results from the queue
        while results_queue.qsize() > 0:
            results.append(results_queue.get())
        # Save results to disk
        with open("results.json", "w") as f:
            json.dump(results, f)
        # Wait a bit
        time.sleep(5)

if __name__ == "__main__":
    main()
