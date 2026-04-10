"""
This script helps debug the KeyError: 'users' in relbench DB construction.
It tries to load the rel-avito dataset and prints the tables present in db.table_dict.
"""
from relbench.datasets import get_dataset
from relbench.tasks import get_task

def debug_relbench_db(dataset_name="rel-stack", task_name="post-votes"):
    dataset = get_dataset(dataset_name)
    print(f"Loaded dataset: {dataset}")
    try:
        db = dataset.get_db(upto_test_timestamp=False)
        print(f"DB object created: {db}")
        print(f"Tables in db.table_dict: {list(db.table_dict.keys())}")
    except Exception as e:
        print(f"Exception during db construction: {e}")
        raise

if __name__ == "__main__":
    debug_relbench_db()
