### Webscraping
You can scrape web pages to text documents in order to use them as documents for chroma. The web scraping uses playwright and requires that the web engines are installed. After starting the virtual env run:</BR>

```
playwright install
```

The web scraping is prepared with config files in web_scrape_configs folder. The format is in json. See the example files for the specfics. The current impelementation is unoptimized, so use with caution for a large number of pages. See the example web_scrape_configs for config format. This will scrape the given web pages and format into a single text document.</BR>

To run the scrape run:
```
python -m document_parsing.web_scraper</BR>
```

Optional param         | Description
---------------------- | -------------
--data-directory       | The directory where your text files are stored. Default "./documents/skynet"
--collection-name      | The name of the collection. Default "skynet"
--web-scrape-directory | The config file to be used for the webscrape. Default "./web_scrape_configs/"