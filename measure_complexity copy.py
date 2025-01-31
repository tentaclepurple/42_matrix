import psutil
import matplotlib.pyplot as plt
import tracemalloc
import gc
import time

from test_utility import vector_utility, matrix_utility


""" def utility(size):

   # Create vector with specified size
   v = Vector([float(i) for i in range(size)])
   
   # Calculate dimensions for the most square matrix possible
   dim = int(size ** 0.5)
   if dim * dim < size:
       dim += 1
       
   # Convert to matrix
   m = v.to_matrix(dim, dim)
   
   # Convert back to vector
   v2 = m.to_vector()
   
   return v2.size() """


def measure_complexity(size):
   """
   Measures both memory and time complexity for a specific input size
   Returns: (peak_memory_kb, execution_time_ms)
   """
   # Force garbage collection before measuring
   gc.collect()
   
   # Start memory tracking
   tracemalloc.start()
   
   # Start time measurement
   start_time = time.time()
   
   # Execute function

   vector_utility(size)
   #matrix_utility(size)


   # Get execution time
   execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
   
   # Get peak memory
   current, peak = tracemalloc.get_traced_memory()
   
   # Stop memory tracking
   tracemalloc.stop()
   
   return peak / 1024, execution_time  # Memory in KB, time in ms


def analyze_complexity():
   # Test sizes (exponential increase for better visualization)
   sizes = [4, 16, 64, 256, 1024, 4096]
   memory_usage = []
   time_usage = []
   
   print("Analyzing space and time complexity...")
   print("Input size | Memory (KB) | Time (ms)")
   print("-" * 40)
   
   for size in sizes:
       mem, time_taken = measure_complexity(size)
       memory_usage.append(mem)
       time_usage.append(time_taken)
       print(f"{size:^10} | {mem:^10.2f} | {time_taken:^8.2f}")
   
   # Create plot with two y-axes
   fig, ax1 = plt.subplots(figsize=(10, 6))
   
   # Plot memory usage
   ax1.plot(sizes, memory_usage, 'bo-', label='Memory')
   ax1.set_xlabel('Input size (n)')
   ax1.set_ylabel('Memory usage (KB)', color='b')
   ax1.tick_params(axis='y', labelcolor='b')
   
   # Create second y-axis for time
   ax2 = ax1.twinx()
   ax2.plot(sizes, time_usage, 'ro-', label='Time')
   ax2.set_ylabel('Execution time (ms)', color='r')
   ax2.tick_params(axis='y', labelcolor='r')
   
   # Set logarithmic scale for better visualization
   ax1.set_xscale('log')
   ax1.set_yscale('log')
   ax2.set_yscale('log')
   
   # Add title and grid
   plt.title('Space and Time Complexity Analysis')
   ax1.grid(True)
   
   # Add legend
   lines1, labels1 = ax1.get_legend_handles_labels()
   lines2, labels2 = ax2.get_legend_handles_labels()
   ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
   
   plt.show()
   
   # Analyze growth ratios for both memory and time
   print("\nAnalysis results:")
   
   # Memory complexity analysis
   memory_ratios = []
   for i in range(1, len(memory_usage)):
       ratio = (memory_usage[i] / memory_usage[i-1]) / (sizes[i] / sizes[i-1])
       memory_ratios.append(ratio)
   
   avg_memory_ratio = sum(memory_ratios) / len(memory_ratios)
   print("Memory complexity:", end=" ")
   if avg_memory_ratio <= 1.2:
       print("Appears to be O(n) - Linear")
   elif avg_memory_ratio <= 2.2:
       print("Appears to be O(n²) - Quadratic")
   else:
       print(f"Appears to be O(n^{avg_memory_ratio:.1f})")
   
   # Time complexity analysis
   time_ratios = []
   for i in range(1, len(time_usage)):
       ratio = (time_usage[i] / time_usage[i-1]) / (sizes[i] / sizes[i-1])
       time_ratios.append(ratio)
   
   avg_time_ratio = sum(time_ratios) / len(time_ratios)
   print("Time complexity:", end=" ")
   if avg_time_ratio <= 1.2:
       print("Appears to be O(n) - Linear")
   elif avg_time_ratio <= 2.2:
       print("Appears to be O(n²) - Quadratic")
   else:
       print(f"Appears to be O(n^{avg_time_ratio:.1f})")


if __name__ == "__main__":
   analyze_complexity()
