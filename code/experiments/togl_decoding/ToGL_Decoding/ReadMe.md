# CLI
Details of the `togl_generate` python CLI are listed below:
|Parameter (Short) | Description | Required (Default) |
|------------------|-------------|--------------------|
|--intput (-i) | Input csv file path | Yes |
|--src-col (-sc) | Column name in input containing source texts | Yes |--togl-left (-tl) | File path to json dictionary of togl distributions for left summaries | Yes |
|--togl-right (-tr) | File path to json dictionary of togl distributions for right summaries | Yes |
|--output (-o) | Output csv path to store generated summaries | Yes |
|--model (-m) | Base pretrained model checkpoint to use in togl decoding | No ('facebook/bart-large-xsum') |
|--togl-func (-tf) | Function to use in combining output word distributions and ToGL distributions | No ('sum') |
|--togl-start (-ts) | Minimum index of token when togl distributions are incorporated | No (3) |
| --togl-weight (-tw) | Weighting of togl distributions in generation (should be less than 1) | No (0.1) |
| --num-beams (-nb) | Number of beams to use in beam search during generation | No (3) |
| --no-repeat-ngram (-nr) | No Repeat Ngram Size to constrain generations | No (3) |
| --min-length (-mi) | Minimum length of generations | No (16) |
| --max-length (-ma) | Maximum length of generations | No (128) |
|--device (-d) | Torch cuda device to use in generation | No ('cuda:0') |
