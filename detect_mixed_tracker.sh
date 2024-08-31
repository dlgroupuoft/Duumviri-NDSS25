#!/bin/bash 

cd code
./clean.sh
echo "Working on $1"
# Step 1, crawl the page to produce a pagedelta file. Extracted pagedeltas will be saved to ./parameter_pagedelta folder.
python3 crawl.py 2 $1


# Step 2, extract features from the pagedelta. This will create a folder named parameter_pagedelta_tmp_results with one csv for each page delta file.
python3 pagedelta.py ./parameter_pagedelta/ 0 0 -1

# Step 3, invoke the modles. The script will print out all analysis results. The format is one url result pair per line. Like
python3 eval.py eval ./parameter_pagedelta_tmp_results/
# URL parameter analyasis_result
# URL parameter analyasis_result
# ...

# rm -rf ./parameter_pagedelta/ ./parameter_pagedelta_tmp_results/