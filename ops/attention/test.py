from query_pruning import run_test as query_run_test
from query_head_group_pruning import run_test as query_head_group_run_test
from query_head_pruning import run_test as query_head_run_test

if __name__ == '__main__':
    print(f"Run query pruning test...")
    query_run_test()

    print(f"Run query head group pruning test...")
    query_head_group_run_test()

    print(f"Run query head pruning test...")
    query_head_run_test()
