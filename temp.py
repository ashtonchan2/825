import pstats

p = pstats.Stats("stats.txt")
p.strip_dirs().sort_stats("cumulative").print_stats(20)
p.print_callers(20, "__getitem__")