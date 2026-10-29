# Hand-written evaluation requests (written by hand, not by gen_data.py). Writes handwritten_eval.jsonl.
import json
A = '["all"]'
def u(url, fields): return '{"url":"%s","extract":%s}' % (url, fields)
E_SCOPE = '{"error":"out_of_scope"}'
E_NOURL = '{"error":"no_url"}'
ROWS = [
 ("go to example.com and give me the data", u("https://example.com", A)),
 ("Go to example.com and give me the data", u("https://example.com", A)),
 ("go to https://example.com and give me the links", u("https://example.com", '["links"]')),
 ("what's the title of example.org?", u("https://example.org", '["title"]')),
 ("scrape iana.org/domains and get the headings", u("https://iana.org/domains", '["headings"]')),
 ("pull all the links from news.ycombinator.com", u("https://news.ycombinator.com", '["links"]')),
 ("grab the images on unsplash.com", u("https://unsplash.com", '["images"]')),
 ("i need the tables from en.wikipedia.org/wiki/List_of_countries", u("https://en.wikipedia.org/wiki/List_of_countries", '["tables"]')),
 ("get me the text of quotes.toscrape.com", u("https://quotes.toscrape.com", '["text"]')),
 ("books.toscrape.com - give me everything", u("https://books.toscrape.com", A)),
 ("open http://httpbin.org/html and extract the paragraphs", u("http://httpbin.org/html", '["text"]')),
 ("scrape the meta description of stripe.com", u("https://stripe.com", '["meta"]')),
 ("visit vercel.com and grab the headings and links", u("https://vercel.com", '["headings","links"]')),
 ("fetch developer.mozilla.org", u("https://developer.mozilla.org", A)),
 ("what's on kotlinlang.org", u("https://kotlinlang.org", A)),
 ("extract the page title and the images from pexels.com", u("https://pexels.com", '["title","images"]')),
 ("can you scrape weather.com for me", u("https://weather.com", A)),
 ("crawl docs.docker.com/get-started and give me the text", u("https://docs.docker.com/get-started", '["text"]')),
 ("get all urls on https://www.gov.za", u("https://www.gov.za", '["links"]')),
 ("use puppeteer to open sars.gov.za and get the headings", u("https://sars.gov.za", '["headings"]')),
 ("I want the photos from flickr.com/explore", u("https://flickr.com/explore", '["images"]')),
 ("scrape https://www.timeanddate.com/worldclock for the tables", u("https://www.timeanddate.com/worldclock", '["tables"]')),
 ("give me the seo metadata of shopify.com", u("https://shopify.com", '["meta"]')),
 ("go to EXAMPLE.COM and get the title", u("https://example.com", '["title"]')),
 ("example.com", u("https://example.com", A)),
 ("dump https://httpbin.org", u("https://httpbin.org", A)),
 ("pull the h1s and paragraphs off w3.org", u("https://w3.org", '["headings","text"]')),
 ("go to archive.org and list the links", u("https://archive.org", '["links"]')),
 ("Hey, could you go to mozilla.org and fetch the page title?", u("https://mozilla.org", '["title"]')),
 ("scrape craigslist.org/about for text and links", u("https://craigslist.org/about", '["text","links"]')),
 ("get the tables and headings from wikipedia.org", u("https://wikipedia.org", '["headings","tables"]')),
 ("show me the images and links on dribbble.com/shots", u("https://dribbble.com/shots", '["links","images"]')),
 ("read kubernetes.io/docs and give me the contents", u("https://kubernetes.io/docs", A)),
 ("whats on lichess.org?", u("https://lichess.org", A)),
 ("extract everything from https://www.capetown.gov.za/", u("https://www.capetown.gov.za/", A)),
 ("get the text from medium.com/about please", u("https://medium.com/about", '["text"]')),
 ("go to tailwindcss.com and give me the headings, links and images", u("https://tailwindcss.com", '["headings","links","images"]')),
 ("scrape the website and give me the links", E_NOURL),
 ("go to the site and get the data", E_NOURL),
 ("extract the tables from that page", E_NOURL),
 ("crawl my website", E_NOURL),
 ("what's the weather in durban", E_SCOPE),
 ("write a haiku about the sea", E_SCOPE),
 ("send an email to info@example.com", E_SCOPE),
 ("who is the president of south africa", E_SCOPE),
 ("tell me a joke about programmers", E_SCOPE),
 ("how do i center a div", E_SCOPE),
 ("good morning", E_SCOPE),
 ("calculate 15% of 200", E_SCOPE),
 ("book me a table for two tonight", E_SCOPE),
]
with open("handwritten_eval.jsonl", "w", newline="\n") as f:
    for i, o in ROWS:
        json.loads(o)
        f.write(json.dumps({"input": i, "output": o}) + "\n")
print(len(ROWS), "rows")
