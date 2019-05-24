import os
from scripts import data_processing as dp

dp.process_data ("2003_np.txt", 64)



os.system('perl Eval-Score-3.0.pl test_input.txt test_score.txt test_out.txt 0')