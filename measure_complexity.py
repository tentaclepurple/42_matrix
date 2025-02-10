import psutil
import matplotlib.pyplot as plt
import tracemalloc
import gc
import time
from complexity_tests import *


def measure_complexity(func, size):
	gc.collect()
	tracemalloc.start()

	start_time = time.time()
	func(size)
	execution_time = (time.time() - start_time) * 1000 

	current, peak = tracemalloc.get_traced_memory()
	tracemalloc.stop()

	return peak / 1024, execution_time 


def analyze_complexity():
	# Test sizes (exponential increase for better visualization)
	sizes = [4, 16, 64, 256, 1024, 4096]

	functions = [
			vector_utility,
			vector_add_complexity,
			vector_sub_complexity,
			matrix_add_complexity,
			matrix_add_complexity,
			matrix_scalar_complexity,
			vector_linear_combination_complexity,
			linear_interpolation_vector_complexity,
			dot_product_complexity,
			norms_complexity,
			angle_cos_complexity,
			matrix_vector_mult_complexity,
			transpose_complexity,
			row_echelon_complexity,
			determinant_complexity,
			inverse_complexity,
			rank_complexity
	]
	memory_usage = []
	time_usage = []

	print("Analyzing space and time complexity...")

	for func in functions:
		print(f"Function: {func.__name__}")
		print("Input size | Memory (KB) | Time (ms)")
		print("-" * 40)

		results = list(map(lambda size: measure_complexity(func, size), sizes))

		memory_usage, time_usage = zip(*results)

		for size, mem, time_taken in zip(sizes, memory_usage, time_usage):
			print(f"{size:^10} | {mem:^10.2f} | {time_taken:^8.2f}")

		print("\n")


		fig, ax1 = plt.subplots(figsize=(10, 6))
		ax1.plot(sizes, memory_usage, 'bo-', label='Memory')
		ax1.set_xlabel('Input size (n)')
		ax1.set_ylabel('Memory usage (KB)', color='b')
		ax1.tick_params(axis='y', labelcolor='b')

		ax2 = ax1.twinx()
		ax2.plot(sizes, time_usage, 'ro-', label='Time')
		ax2.set_ylabel('Execution time (ms)', color='r')
		ax2.tick_params(axis='y', labelcolor='r')

		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax2.set_yscale('log')

		plt.title(f'Complexity Analysis of {func.__name__}')
		ax1.grid(True)

		lines1, labels1 = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

		plt.show()

		print("Analysis results:")

		memory_ratios = [(memory_usage[i] / memory_usage[i-1]) / (sizes[i] / sizes[i-1]) 
						 for i in range(1, len(memory_usage))]
		avg_memory_ratio = sum(memory_ratios) / len(memory_ratios)
		print("Memory complexity:", "O(n)" if avg_memory_ratio <= 1.2 else 
			  "O(n²)" if avg_memory_ratio <= 2.2 else f"O(n^{avg_memory_ratio:.1f})")

		time_ratios = [(time_usage[i] / time_usage[i-1]) / (sizes[i] / sizes[i-1]) 
					   for i in range(1, len(time_usage))]
		avg_time_ratio = sum(time_ratios) / len(time_ratios)
		print("Time complexity:", "O(n)" if avg_time_ratio <= 1.2 else 
			  "O(n²)" if avg_time_ratio <= 2.2 else f"O(n^{avg_time_ratio:.1f})")

		print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
   analyze_complexity()
