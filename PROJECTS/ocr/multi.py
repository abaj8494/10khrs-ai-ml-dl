import multiprocessing
import threading
import time
import os

def square_worker(x):
    """
    Compute x*x, print the result along with thread name and process ID.
    """
    # Optional small delay to simulate "work"
    time.sleep(0.01)
    result = x * x
    print(f"PID {os.getpid()} - Thread {threading.current_thread().name} "
          f"computed square of {x} as {result}")

def run_chunk(chunk_id, tasks):
    """
    Run a chunk of tasks in the current process, each task handled by a thread.
    """
    print(f"Process {chunk_id} started (PID={os.getpid()}). Handling {len(tasks)} tasks.")
    threads = []
    
    # Create a thread for each task
    for x in tasks:
        t = threading.Thread(target=square_worker, args=(x,))
        t.start()
        threads.append(t)
    
    # Wait for all threads to finish
    for t in threads:
        t.join()
    
    print(f"Process {chunk_id} finished (PID={os.getpid()}).")

def main():
    # We want 1600 tasks (0..1599)
    total_tasks = 1600
    tasks = list(range(total_tasks))
    
    # 8 processes, each will handle 200 tasks
    num_processes = 8
    chunk_size = total_tasks // num_processes

    processes = []
    print("Main process starting 8 worker processes.")
    
    for chunk_id in range(num_processes):
        # Determine which portion of tasks goes to this process
        start_index = chunk_id * chunk_size
        end_index = (chunk_id + 1) * chunk_size
        chunk_tasks = tasks[start_index:end_index]
        
        # Create the process
        p = multiprocessing.Process(target=run_chunk, args=(chunk_id, chunk_tasks))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    print("All worker processes finished.")

if __name__ == "__main__":
    main()

