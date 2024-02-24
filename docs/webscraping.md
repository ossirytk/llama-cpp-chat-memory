### Webscraping
You can scrape web pages to text documents in order to use them as documents for chroma. 

Optional. The old web scraping uses playwright and requires that the web engines are installed. After starting the virtual env run:</BR>

```
playwright install
```

The web scraping is prepared with config files in web_scrape_configs folder. The format is in json. See the example files for the specfics. A number of regex filters are used to clean the scrape data. You can modify and add filters if you want. The filters are stored in the src/llama_cpp_chat_memory/run_files/filters/web_scrape_filters.json file.</BR>

To run the scrape run:
```
python -m document_parsing.web_scraper</BR>
```

Optional param         | Description
---------------------- | -------------
--data-directory       | The directory where your text files are stored. Default "./run_files/documents/skynet"
--collection-name      | The name of the collection. Default "skynet"
--web-scrape-directory | The config file to be used for the webscrape. Default "./run_files/web_scrape_configs/"