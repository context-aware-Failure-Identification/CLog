# CLog: Context-aware method for log-based failure identification
This project provides code for the paper "Failure Identification from Unstable Log Data using Deep Learning"

## Quick Start

CLog requires log data as input. The minimal requirements for the inputs are 1) log messages and 2) timestamp. Additionally, we assume the availability of a task ID to group logs.
Nevertheless, this assumption can be dropped by including multi-index grouping of the log events by time (e.g., hours and seconds windows.).
For evaluation, it's important to know additionally the severity degree of the individual label of the window. Further information like service ID, workload ID, process ID, log level, parameters, templates etc, are available and can be used for greater in-depth information.  

For parsing one can use Drain. We provide an in-memory implementation of the parser. However, if using the parser please cite the relevant authors: Pinjia He et al. https://github.com/logpai/logparser/tree/master/logparser/Drain. 

### Prerequisite
The requirements can be found in the requirements.tex

1. Clone the Repository git clone https://github.com/context-aware-Failure-Identification/CLog.git

2. Install the requirements using: pip install -r /path/to/requirements.txt

3. Customize the directories and paths

4. Run the parser in the Drain.py: 
   1. Set the depth and the similarity threshold parameters

5. Run the preprocessing scripts 1_, 2_, 3_ to prepare the data;
   1. Set the parameter window_size
   2. Set the training and validation sizes to tune the hyperparameters

6. Run CLog_main.py to extract context-aware subprocesses;
   1. The parameters of the CLog are set here;
   2. They are set at initialized as described in the paper;

7. Run FD.py to detect failures
   1. Here one sets the number of states

8. Run FTI.py to identify failure types
   1. Adjust the training models for the different number of failure types
   2. Tip: Use different weights if the distributions of your classes are severe.

------------
## Data
The processed data can be found in: https://tubcloud.tu-berlin.de/s/wNTbFW5wfWxqpCH (Due to github limmitation of 25 MB large files). The corresponding templates obtained from Drain are stored in OpenStackTemplates.csv. You can put your data when your experiments for easier navigation. 
