_Note: All code was written based on past coding experience and self-written libraries. No third-party code was copied or used as reference for the code present in this directory._
# Data Collection

Data collection was performed in  steps, each in its own jupyter notebook in the directory:
1) FlipSideScraping
2) Adding Expert Opinions
3) URL Cleaning
4) Reddit Scraping
5) Twitter Scraping
6) Consolidation

## 1 - FlipSide Scraping
TheFlipSide [theflipside.io](https://www.theflipside.io/) provides daily publications of articles on current events. 
Each article contains an overview using quotes from news articles, left and right perspective opinions, and expert opinions from news articles.
As a base for the PoliSum dataset, each article from TheFlipSide was scraped. Because TheFlipSide does not offer an API, pages were crawled manually using the `Selenium` library and `Beautiful Soup`.

TheFlipSide articles are each loaded dynamically with a custom url for each article. For this reason, Selenium was necessary in order to dynamically load pages, and articles had to be fetched from the search page.
The following steps were taken as attempts to collect all TheFlipSide summaries and news links:
1) TheFlipSide contains a page with an archive of articles from the last couple months. All links on this page were scraped and stored.
2) To collect the remaining articles, queries of the form `MONTH YEAR` were created and used with TheFlipSide search page. Each query returns a few dozen results.
3) Finally, some dates were still missed by the above two steps. As an attempt to fill in the remaining gaps, more specific queries in the form `MONTH DAY, YEAR` and the scraping process was repeated.

All relevant code is included in `1 - FlipSide Scraping.ipynb` notebook.

## 2 - Adding Expert Opinions
In the first scraping pass, expert opinions were not included in the data collection process, but some methods utilize expert opinions in determining contrastive summaries.
So, the stored articles were parsed again to collect the expert opinions on corresponding sides. The relevant code is included in `2 - Adding Expert Opinions.ipynb`.

## 3 - URL Cleaning
The initial TheFlipSide scraping collected as many gold perspective summary pairs as possible. However, source documents from social media still need to be collected and correspond to these articles.
The news links on TheFlipSide articles allow for connecting each article to current events that might be discussed on other platforms. However, the urls may be repeated, irrelevant, or shortened for click tracking.
So, each set of outbound links need to be cleaned by 1) removing duplicates, 2) expanding shortened urls, and 3) truncating the number of urls to select only those appearing near the top of the article.

The code implementing this cleaning is included in `3 - URL Cleaning`

## 4 - Reddit Scraping
Using the cleaned URLs, each url is passed to the PushShift Reddit API to scrape a maximum of 10 related posts mentioning each URL. All posts are scraped from three subreddits: r/politics, r/usnews, and r/worldnews.
The Reddit Scraping is done in two passes:

1) Each outbound link is used as a query to the Reddit API with a date range of 30 days after the article was published. However, this returned few links and in some cases no links for each article.
2) Then, each article title is used as a query to the Reddit API with a tighter date range of 10 days after the article publication date.

## 5 - Twitter Scraping
Similarly to the Reddit Scraping, each url is also used with the Twitter API to collect additional source documents. For each url, a maximum of 10 posts are collected from Twitter, excluding retweets, replies, and tweets from verified accounts.
Verified accounts are removed to avoid tweets that simply use the original news article's headline. The list of urls are created in `5 - Twitter Scraping.ipynb` and passed to the `scrape_twitter.sh` bash script to iterate over twitter queries.
Scraping is performed using a Twitter Developer Academic license and the [pull_twitter repository](https://github.com/dhudsmith/pull_twitter) I created with my undergraduate research advisor. The configuration file (with bearer token omitted) is also included in the abs_config.yaml file that was passed to pull_twitter.

After the scraping is completed, all retrieved tweets are collected, filtering out any tweets that simply contain a url, removing usernames, and removing repeated phrases that may indicate article titles.

## 6 - Consolidation
Finally, after all scraping is completed, all data from TheFlipSide, Reddit, and Twitter are combined into a single file making up the PoliSum dataset.
This is a simple step, merely involving combining the data from three sources based on an aligned title column. The code is included in the `6 - Consolidation.ipynb` notebook.
