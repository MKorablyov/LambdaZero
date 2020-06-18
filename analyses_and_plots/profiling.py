import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
