#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu
import requests
from bs4 import BeautifulSoup
import time
import re
import os
import argparse
import csv
"""
search keyword = term in https://pubmed.ncbi.nlm.nih.gov/
each page has 10 results
"""

def read_file(FILEPATH):
    with open(FILEPATH,'r') as f:
      file_text = f.read().strip()
      f.close()
    return file_text

def get_html(url,header,params):
    """
    :param url: the link where we want to scrape information
    :param header: the request header
    :param params: a dictionary = {"term": "keyword"}, the keyword/term we want to search
    :return: a soup object
    """
    if params is None:
        webpage = requests.get(url, headers=header)
    else:
        webpage = requests.get(url, params=params, headers=header)
    # r.encoding = 'utf-8'

    html = webpage.text

    soup = BeautifulSoup(html, "lxml")
    return soup


def get_result_pages(url, header, term):
    """

    :param url: the link where we want to scrape information
    :param header: the request header
    :param search_params: a dictionary = {"term": "keyword"}, the keyword/term we want to search
    :return: int, number of pages that contains the results
    """
    search_params = {"term": term}
    search_soup = get_html(url, header, search_params)


    results = search_soup.select(
        '#search-results > div.top-wrapper > div.results-amount-container > div.results-amount > span')  # this returns to a list
    results = results[0].text
    if len(results) != 0:
        new_results = int(str(results).replace("\n", "").replace(",", ""))
        if new_results % 10 == 0:
            pages = new_results / 10
        else:
            pages = int(new_results / 10) + 1
        print(f"we have found {pages} pages results related to the term {term}")
    else:
        pages = 0
        print(f"No results related to the term {term}")
    return pages







def web_scrape(csv_file_name,start_page_enter,end_page,url,header,term):
    csv_file = open(csv_file_name, 'w')
    csv_writer = csv.writer((csv_file))
    csv_writer.writerow(['PMID','Title','Abstract','Keyword','Year','Link','DOI'])
    for page in range(start_page_enter,end_page,1):
        params = {"term":term,"page":str(page)}
        result_soup = get_html(url,header,params)
        time.sleep(2)
        # need to make sure not only include specific child, including all child
        """ when using inspect - copy-selector
        result = #search-results > section > div.search-results-chunks > div > article:nth-child(5) > div.docsum-wrap > div.docsum-content > a
        article:nth-child(5)--- remove anything after :
        """
        links = result_soup.select('#search-results > section > div.search-results-chunks > div > article > div.docsum-wrap > div.docsum-content > a')

        for ix, link in enumerate(links):
            try:
                print(f"this program is scraping page: {page} and link #{ix}")

                ########## extract PMID
                id_soup = link["data-article-id"]

                if (id_soup is None):
                    time.sleep(2)
                    continue

                elif len(id_soup)<1:
                    time.sleep(2)
                    continue

                else:
                    pmid = id_soup.strip()
                print(f"id {pmid}")


                ########## extract article link
                link_soup = link["href"]

                if link_soup is None:
                    time.sleep(2)
                    continue

                elif len(link_soup) <1:
                    time.sleep(2)
                    continue

                else:
                    article_link = url + link_soup
                print(f"article_link {article_link}")





                #######  request the html for an article, get detail info from the article detail webpage
                article_soup = get_html(article_link, header, None)
                time.sleep(1)

                ######### extract article title

                title_soup = article_soup.select('#full-view-heading > h1')

                if title_soup is None:
                    title = "Missing"
                elif len(title_soup)<1:
                    title = "Missing"
                else:
                    title = title_soup[0].text.strip()
                print(f"title: {title}")

                ######## extract article abstract

                abstracts = article_soup.select('#eng-abstract > p')
                if (abstracts is None):
                    print("No abstract available and skip to next article")
                    time.sleep(2)
                    continue

                elif (len(abstracts) == 0):
                    print("No abstract available and skip to next article")
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



                print(f"abstract: {abstract}")

                ####### extract article keywords: #abstract > p

                key_paragraph= article_soup.find('div', class_= 'abstract').find('strong', class_ ='sub-title', string = re.compile('Keywords'))

                if key_paragraph is None:
                    keywords = 'Missing'
                else:
                    keywords = re.sub("\s{2,}", " ", key_paragraph.parent.text.strip())
                print(f"keywords: {keywords}")

                # try:
                #     author_list= article_soup.select('#full-view-heading > div.inline-authors > div > div > span > a')
                #
                #     authors = []
                #     if len(author_list) > 0:
                #         for au in author_list[:min(4, len(author_list))]:
                #             authors.append(au.text)
                # except Exception as e:
                #     authors = "Missing"
                # print(f"authors {authors}")
                #
                # try:
                #     journal = article_soup.select('#full-view-heading > div.article-citation > div.article-source > div > button')[0].text.strip()
                # except Exception as e:
                #     journal = "Missing"
                # print(f"journal {journal}")

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
                print(f"year: {year}")

                doi_soup = article_soup.find("span", class_='identifier doi')
                if (doi_soup is None) :
                    doi = "Missing"
                elif (len(doi_soup)<1):
                    doi = "Missing"
                else:
                    doi = doi_soup.find("a", class_='id-link').text.strip()

                print(f"doi: {doi}")


                csv_writer.writerow([pmid,title,abstract,keywords,year,article_link,doi])

                if ix % 3 == 0:
                    time.sleep(3)
                elif ix % 10 ==0:
                    time.sleep(5)
                else:
                    time.sleep(1)


            except Exception as e:
                print(f"error occure when scraping page #{page} and link #{ix}")
                print(e)
                time.sleep(300)
                pass
    csv_file.close()
    print(f"finish scraping page #{page}")



def parse_args():
    parser=argparse.ArgumentParser(description="Scraping article info from PubMed based on keywords",prog = "PubMed Scraper",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--term_path",type=str, default="./data_preparation/input/combined_terms.txt", help = "Path to the txt file stored keyword term")
    parser.add_argument("--start", type=int,default= 30, help="start to scrape info from this page")
    parser.add_argument("--end", type=int, default= 35, help="stop scraping info at this page")
    parser.add_argument("--output_folder", type=str, default='./data_preparation/temp', help="directory to save the scraped info")
    parser.add_argument("--file", type=str, default='sars_v1.csv', help="file to save the scraped info")
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
    csv_file_name = os.path.join(output_folder,args.file)


    # pass the keyword to the url
    url = "https://pubmed.ncbi.nlm.nih.gov"

    # get header from "inspect" - "network" - "All" - "Name" - "user-agent"
    header = {
        "user-agent":
            # "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3"
    }

    result_pages = get_result_pages(url, header, term)

    end_page = min(result_pages, end_page_enter)

    print(f"start scraping term: {term} from page #{start_page_enter}  to page #{end_page_enter}")
    web_scrape(csv_file_name, start_page_enter, end_page, url, header, term)
    print("Done !!!!!!!!!!!!!!!!")


if __name__ == '__main__':
    main()





