Ensure AWS creds exist

ray up ./dataset.yaml

It takes some time for all nodes to setup even after head node setup is complete
Use ray monitor ./dataset.yaml to check available resources 

ray submit ./dataset.yaml gradient_health.py

** the ray transform code will have 2 progress bars and there will be some time gap between the 2 wherein dynamic shapes + end samples are written by a single worker.


** gradient_health.py can be changed on lines 461, 477 and 487 to adjust number of samples, number of workers and dataset output path respectively
*commenting out line 461 => full dataset generated
* the default value of workers on line 477 is set as 50 and not 96 as I encountered some unpredictable issues with more workers after a bug fix I made
* the issue was:- An error occurred (SlowDown) when calling the PutObject operation (reached max retries: 4): Please reduce your request rate.
* this error is probably due to the chunks being written to the temporary hub dataset




explore.ipynb can be used to explore the dataset

P.S. The dataset reshape operation is expensive and is not being performed (xtransforms create 1 mill sample datasets)
The work around for this is that the dataset generated prints the number of output samples generated,
for example "Size of output dataset is 13996". This can be used to generate a datasetview of the 1 mill sample dataset i.e. dsv = ds[0:13996]
