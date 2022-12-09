#!/bin/bash

CONFIG="C:\Users\ndeas\Desktop\Columbia\Research\Abstract-Contrastive-Opinion-Summarization\code\data_collection\abcos_config.yaml"
INDEX=0

cat "$1" | while  read line || [[ -n $line ]];
do
	python pull_twitter/pull_twitter.py --config-file "$CONFIG" search -q "$line" -mr 10 -st "2018-09-01" 2>&1 | tee -a twitter_scrape_log
	echo "Done with $line" 2>&1 | tee -a twitter_scrape_log
	
	sleep 1

done