# Instructions

We used ClevrGeneration.ipynb to generate tasks and OOD datasets. In particular, for each task, the code refers to task_ID.json inside dataclevr.zip to decide colors and shapes. Besides that, the same code uses a modified version of CLEVR generation code (inside codeclevr.zip)


A further preprocessing step is done with CleanAndSplitClevr.ipynb where overlapped objects are removed from the dataset. Moreover it decides how to split the data into train/test/val sets.

