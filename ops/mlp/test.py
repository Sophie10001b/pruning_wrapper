from bm_pruning import run_test as bm_run_test
from bn_pruning import run_test as bn_run_test
from bk_pruning import run_test as bk_run_test

if __name__ == '__main__':
    print(f"Run M-axis pruning test...")
    bm_run_test()

    print(f"Run N-axis pruning test...")
    bn_run_test()

    print(f"Run K-axis pruning test...")
    bk_run_test()