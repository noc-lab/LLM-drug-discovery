#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu

import requests
from bs4 import BeautifulSoup
import time
import re
import os
import argparse
import csv

def read_file(FILEPATH):
    with open(FILEPATH,'r') as f:
      file_text = f.read().strip()
      f.close()
    return file_text

def get_html(url,header,params):
    if params is None:
        webpage = requests.get(url, headers=header)
    else:
        webpage = requests.get(url, params=params, headers=header)

    html = webpage.text
    soup = BeautifulSoup(html, "lxml")
    return soup


def get_result_pages(url, header, term):
    search_params = {"term": term}
    search_soup = get_html(url, header, search_params)


    results = search_soup.select(
        '#search-results > div.top-wrapper > div.results-amount-container > div.results-amount > span')
    results = results[0].text
    if len(results) != 0:
        new_results = int(str(results).replace("\n", "").replace(",", ""))
        if new_results % 10 == 0:
            pages = new_results / 10
        else:
            pages = int(new_results / 10) + 1
    else:
        pages = 0
    return pages


def web_scrape(csv_file_name,
               start_page_enter,
               end_page,
               url,
               header,
               term):
    csv_file = open(csv_file_name, 'w')
    csv_writer = csv.writer((csv_file))
    csv_writer.writerow(['PMID','Title','Abstract','Keyword','Year','Link','DOI'])

    for page in range(start_page_enter, end_page,1):
        params = {"term": term, "page": str(page)}
        result_soup = get_html(url, header, params)
        time.sleep(2)
        links = result_soup.select('#search-results > section > div.search-results-chunks > div > article > div.docsum-wrap > div.docsum-content > a')

        for ix, link in enumerate(links):
            try:
                id_soup = link["data-article-id"]
                if (id_soup is None):
                    time.sleep(2)
                    continue
                elif len(id_soup)<1:
                    time.sleep(2)
                    continue
                else:
                    pmid = id_soup.strip()

                link_soup = link["href"]
                if link_soup is None:
                    time.sleep(2)
                    continue
                elif len(link_soup) <1:
                    time.sleep(2)
                    continue
                else:
                    article_link = url + link_soup

                article_soup = get_html(article_link, header, None)
                time.sleep(1)

                title_soup = article_soup.select('#full-view-heading > h1')
                if title_soup is None:
                    title = "Missing"
                elif len(title_soup)<1:
                    title = "Missing"
                else:
                    title = title_soup[0].text.strip()

                abstracts = article_soup.select('#eng-abstract > p')
                if (abstracts is None):
                    time.sleep(2)
                    continue

                elif (len(abstracts) == 0):
                    time.sleep(2)
                    continue
                elif len(abstracts) == 1:
                    abstract = abstracts[0].text.strip()
                else:
                    abstract_ls = []
                    for ab in abstracts:
                        paragraph = re.sub("\s{2,}", " ", ab.text.strip())
                        abstract_ls.append(paragraph)
                    abstract = " ".join(abstract_ls)

                key_paragraph= article_soup.find('div', class_= 'abstract').find('strong', class_ ='sub-title', string = re.compile('Keywords'))
                if key_paragraph is None:
                    keywords = 'Missing'
                else:
                    keywords = re.sub("\s{2,}", " ", key_paragraph.parent.text.strip())

                year_soup = article_soup.select('#full-view-heading > div.article-citation > div.article-source > span.cit')
                if (year_soup is None):
                    year = 'Missing'
                elif (len(year_soup)<1):
                    year = 'Missing'
                else:
                    year_text = year_soup[0].text.strip()
                    p = re.compile('\d{4}')
                    m = re.match(p, year_text)
                    if m:
                        year = m.group().strip()
                    else:
                        year = 'Missing'

                doi_soup = article_soup.find("span", class_='identifier doi')
                if (doi_soup is None) :
                    doi = "Missing"
                elif (len(doi_soup)<1):
                    doi = "Missing"
                else:
                    doi = doi_soup.find("a", class_='id-link').text.strip()

                csv_writer.writerow([pmid, title, abstract,keywords,year,article_link,doi])

                if ix % 3 == 0:
                    time.sleep(3)
                elif ix % 10 == 0:
                    time.sleep(5)
                else:
                    time.sleep(1)

            except Exception as e:
                time.sleep(300)
                pass
    csv_file.close()



def parse_args():
    parser=argparse.ArgumentParser(description="Scraping article info from PubMed based on keyword terms", prog = "PubMed Scraper",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--term_path",type=str, default="./data_preparation/input/terms.txt")
    parser.add_argument("--start", type=int,default= 0)
    parser.add_argument("--end", type=int, default= 100)
    parser.add_argument("--output_folder", type=str, default='./data_preparation/input/datasets/Nipah')
    parser.add_argument("--file", type=str, default='Nipah.csv')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    term_path = args.term_path
    term = read_file(term_path)
    start_page_enter = args.start
    end_page_enter = args.end

    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    csv_file_name = os.path.join(output_folder, args.file)

    url = "https://pubmed.ncbi.nlm.nih.gov"

    header = {
        "user-agent":
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3"
    }

    result_pages = get_result_pages(url, header, term)
    end_page = min(result_pages, end_page_enter)
    web_scrape(csv_file_name, start_page_enter, end_page, url, header, term)

if __name__ == '__main__':
    main()





