# CLI Command
Details of the Zero-Shot and Few-Shot/KFold commands are listed below:


## Zero Shot Abstractive Baseline
The following lists parameters for the command `python -m main zhot ...`
|Parameter (Short) | Description | Required (Default) |
|------------------|-------------|--------------------|
|--model (-m) | The name of the base model checkpoint | Yes |
|--test-fp (-tfp) | File path of the test csv file | Yes |
|--batch-size (-bs) | Batch size to use in evaluation | No (4) |
|-results-fp (-rf) | Directory to store logs and model generations | No ('./') |

## Few Shot/KFold Abstractive Baseline
The following lists parameters for the command `python -m main fhot ...`
|Parameter (Short) | Description | Required (Default) |
|------------------|-------------|--------------------|
|--model (-m) | The name of the base model checkpoint | Yes |
|--train-fp (-tr) | File path of the train csv file | Yes |
|--val-fp (-va) | File path of the validation csv file | No |
|--test-fp (-te) | File path of the test csv file | Yes |
|--batch-size (-bs) | Batch size to use in training and evaluation | No (4) |
|--chkpt-dir (-cd) | Directory to store model checkpoints in during training | No ('./') |
|-results-fp (-rf) | Directory to store logs and model generations | No ('./') |
