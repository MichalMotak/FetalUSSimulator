# import threading

# class TimeoutException(Exception):
#     pass

# def timeout_handler():
#     raise TimeoutException("Function execution timed out!")

# def run_with_timeout(func, args=(), kwargs={}, timeout=10):
#     # Start the timer
#     timer = threading.Timer(timeout, timeout_handler)
#     timer.start()
    
#     try:
#         # Run the function
#         result = func(*args, **kwargs)
#     except TimeoutException:
#         return None  # Or handle timeout as needed
#     finally:
#         # Cancel the timer if the function completes in time
#         timer.cancel()
    
#     return result

# # Example usage:
# import time

# def long_running_function():
#     time.sleep(15)  # This simulates a long-running task
#     return "Completed"

# result = run_with_timeout(long_running_function, timeout=10)
# if result is None:
#     print("Function timed out!")
# else:
#     print("Function returned:", result)








#################################################


from tqdm import tqdm
import time

st = time.time()
ttr = 5
endt = st + ttr


for i in tqdm(range(0,1000000000)):
    if (time.time()>endt):
        print("break")
        break

    pass