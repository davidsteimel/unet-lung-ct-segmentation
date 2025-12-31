import pytest
import glob
import os

TEST_PATH = "data_processed/test/image/"
VAL_PATH = "data_processed/val/image/"
TRAIN_PATH = "data_processed/train/image/"

def get_patient_id(filepath):
    filename = os.path.basename(filepath) 
    patient_id = filename.split('_')[0] 
    return patient_id

def test_no_data_leakage():
    train_files = glob.glob(os.path.join(TRAIN_PATH, '*.jpg'))
    val_files = glob.glob(os.path.join(VAL_PATH, '*.jpg'))
    test_files = glob.glob(os.path.join(TEST_PATH, '*.jpg')) 

    assert len(train_files) > 0, "Training folder is empty"
    assert len(val_files) > 0, "Validation folder is empty"
    assert len(test_files) > 0, "Test folder is empty"

    train_ids = set([get_patient_id(f) for f in train_files])
    val_ids = set([get_patient_id(f) for f in val_files])
    test_ids = set([get_patient_id(f) for f in test_files])

    train_val_overlap = train_ids.intersection(val_ids)
    assert len(train_val_overlap) == 0, f"Error: ID overlap {train_val_overlap}"
    train_test_overlap = train_ids.intersection(test_ids)
    assert len(train_test_overlap) == 0, f"Error: ID overlap {train_test_overlap}"
    val_test_overlap = val_ids.intersection(test_ids)
    assert len(val_test_overlap) == 0, f"Error: ID overlap {val_test_overlap}"