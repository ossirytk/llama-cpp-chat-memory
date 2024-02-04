### Webscraping
You can scrape web pages to text documents in order to use them as documents for chroma. The web scraping uses playwright and requires that the web engines are installed. After starting the virtual env run:</BR>

```
playwright install
```

The web scraping is prepared with config files in web_scrape_configs folder. The format is in json. See the example files for the specfics. The current impelemntation is unoptimized, so use with caution for a large number of pages.</BR>

To run the scrape run:
```
python -m document_parsing.web_scraper</BR>
```
See --help for params