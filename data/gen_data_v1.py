"""Version 1 of the MarvijoWebSearch data generator (commit 1293830), kept to retrain the v1 lane.

Synthetic dataset for MarvijoWebSearch.

MarvijoWebSearch does ONE job: read a request such as "go to example.com and give me the data"
and write a JSON plan that a Puppeteer script runs:

    {"url":"https://example.com","extract":["all"]}

Every example is built from parts (verbs, field words, URLs, noise), so the model must
learn the pattern, not memorise sentences. Three test sets measure generalisation:

- test_unseen_domains : training phrasings, but domain names built from a held-out word list
- test_unseen_phrasings: held-out sentence templates the model never saw
- handwritten_eval    : sentences written by hand, not by this generator (handwritten_eval.jsonl)

The word "example" never appears in the training data, so the demo request about
example.com is a real unseen-domain test.

Usage: python gen_data.py --out . --train 200000
"""
import argparse
import json
import os
import random
import re

FIELDS = ["title", "meta", "headings", "text", "links", "images", "tables"]

FIELD_WORDS = {
    "title": ["title", "page title", "the title tag", "site title", "name of the page", "page name"],
    "meta": ["meta tags", "metadata", "meta description", "seo info", "meta info", "description tag"],
    "headings": ["headings", "headers", "h1s", "section titles", "h1 and h2 tags", "subheadings", "heading tags"],
    "text": ["text", "body text", "paragraphs", "copy", "article text", "main text", "written content", "plain text", "wording"],
    "links": ["links", "urls", "hyperlinks", "hrefs", "anchors", "link list", "outbound links", "nav links"],
    "images": ["images", "pictures", "photos", "image urls", "img tags", "pics", "image sources"],
    "tables": ["tables", "table data", "tabular data", "data tables", "table rows", "html tables"],
}

GENERIC = [
    "the data", "all the data", "the contents", "the content", "everything", "everything on it",
    "all of it", "what's on it", "whats on it", "the page", "the whole page", "all the info",
    "all information", "the info", "a full dump", "the site contents", "its contents", "the page data",
    "a summary of the page", "what it says", "the website data", "all the details", "the page content",
    "all page content", "the raw data", "a full extraction", "the html contents", "every bit of data",
]

# Sentence templates. {u} = url, {f} = field phrase, {g} = generic phrase.
TRAIN_FIELD_T = [
    "go to {u} and get {f}", "go to {u} and give me {f}", "open {u} and extract {f}",
    "scrape {f} from {u}", "extract {f} from {u}", "get {f} from {u}", "grab {f} off {u}",
    "pull {f} from {u}", "visit {u} and pull {f}", "can you go to {u} and get {f}",
    "could you open {u} and grab {f}", "i need {f} from {u}", "i want {f} from {u}",
    "please scrape {u} for {f}", "{u} - get {f}", "{u}: extract {f}", "list {f} on {u}",
    "show me {f} on {u}", "fetch {f} from {u}", "collect {f} from {u}", "navigate to {u} and collect {f}",
    "head over to {u} and get {f}", "check {u} and give me {f}", "load {u} then extract {f}",
    "browse to {u} and copy {f}", "give me {f} from {u}", "get me {f} on {u} please",
    "hey, scrape {u} and return {f}", "use puppeteer to open {u} and extract {f}",
    "launch a browser, go to {u}, get {f}", "what are {f} on {u}", "find {f} on {u}",
    "read {u} and list {f}", "return {f} for {u}", "dump {f} from {u}", "crawl {u} for {f}",
    "can u get {f} from {u}", "yo grab {f} from {u}", "need {f} off {u} asap", "{f} from {u} please",
    "{f} on {u}?", "go on {u} and fetch {f}", "hit {u} and scrape {f}", "parse {f} out of {u}",
    "what is {f} of {u}", "what's {f} of {u}", "tell me {f} of {u}", "scrape {u} for {f}",
]
TRAIN_GENERIC_T = [
    "go to {u} and give me {g}", "go to {u} and get {g}", "scrape {u}", "scrape {g} from {u}",
    "extract {g} from {u}", "open {u} and extract {g}", "visit {u} and grab {g}", "get {g} from {u}",
    "fetch {u}", "can you scrape {u} for me", "i need {g} of {u}", "pull {g} off {u}",
    "what's on {u}", "whats on {u}?", "read {u} for me", "{u}", "{u} - give me {g}",
    "load {u} and return {g}", "use puppeteer to scrape {u}", "browse {u} and give me {g}",
    "please extract {u}", "show me {g} of {u}", "crawl {u}", "dump {u}", "check out {u} and get {g}",
    "head to {u} and pull {g}", "open up {u}", "get me {g} on {u}", "hey can you grab {g} from {u}",
    "navigate to {u}, extract {g}", "tell me what {u} says", "go on {u} and collect {g}",
]
# Held out: only used in test_unseen_phrasings.
TEST_FIELD_T = [
    "would you mind fetching {f} that {u} has", "kindly retrieve {f} located at {u}",
    "point the browser at {u}; i only care about {f}", "harvest {f} across {u}",
    "{u} -> {f}", "i'm after {f}, site is {u}", "take a look at {u} and note down {f}",
    "rip {f} off {u}", "gather up {f} at {u}", "scan {u}, report {f}",
]
TEST_GENERIC_T = [
    "would you mind fetching {g} that {u} has", "kindly retrieve {g} located at {u}",
    "point the browser at {u} and bring back {g}", "harvest {g} across {u}", "{u} -> {g}",
    "take a look at {u} and note down {g}", "rip {g} off {u}", "gather up {g} at {u}", "scan {u}",
    "summarize {u} for me",
]

NO_URL_T = [
    "scrape the website and give me {f}", "go to the site and get {g}", "extract {f} from the page",
    "open the website and pull {g}", "scrape it", "get {f} from that site", "go to the link i sent and get {g}",
    "visit the homepage and grab {f}", "can you scrape {g} from a website", "fetch the page", "extract {g}",
    "scrape {f} from my site", "go to the url and give me {g}", "crawl the website", "get {g} from the web page",
]

OUT_OF_SCOPE = [
    "what's the weather today", "write me a poem about cats", "tell me a joke", "how do i cook rice",
    "translate hello into french", "what is 2 + 2", "who won the world cup in 2010", "hello", "hi there",
    "how are you", "write a python function to sort a list", "summarize this paragraph for me",
    "send an email to {e}", "email {e} the report", "book a flight to cape town", "set a timer for 5 minutes",
    "what time is it", "play some music", "explain quantum computing", "give me a recipe for pancakes",
    "what is the capital of france", "write a cover letter", "help me with my homework", "thanks",
    "turn off the lights", "order a pizza", "what's your name", "delete my files", "reboot the server",
    "call mom", "remind me to buy milk", "draw a picture of a dog", "convert 10 usd to zar",
    "what is love", "sing a song", "how tall is mount everest", "fix this bug in my code",
    "generate a password", "who are you", "log into my bank account", "buy bitcoin", "ok", "yes", "no",
    "write a story about a dragon", "what should i eat tonight", "message {e} that i am late",
    "is it going to rain tomorrow", "how many legs does a spider have", "make me a todo list",
    "can you hear me", "do my taxes", "rate my outfit", "count to ten", "what's the meaning of life",
]

PREFIX = ["", "", "", "", "", "please ", "hey ", "ok ", "quick one: ", "yo ", "hi, ", "agent, ", "scout, ", "task: "]
SUFFIX = ["", "", "", "", "", " please", " thanks", " thx", ".", "!", " asap", " for me", " pls", "?", " now", " :)"]

# Word lists for domains. Train and test lists do not overlap.
TRAIN_WORDS = """
apple river stone cloud pixel orbit maple cedar falcon tiger lemon mango ocean forest summit
rocket planet silver golden coffee bread pizza garden market studio design media news daily
tech code data dev labs cloud store shop books music video photo travel health fitness
finance money bank crypto sport football chess games play kids family home house city metro
green blue red black white bright smart quick fast happy lucky sunny star moon sun fire water
earth wind snow rain storm wave sand rock hill lake bay port bridge tower castle king queen
north south east west urban rural local global world nation africa cape joburg durban zulu
alpha beta gamma delta omega nova zen flow sync link node grid core base hub spot box kit
craft maker build forge works factory lab space zone land field farm fresh organic pure clean
paper ink print press post mail chat talk voice radio tv film cinema art gallery museum school
learn study tutor class course academy campus uni college jobs work career talent hire team
""".split()
TEST_WORDS = """
example quartz velvet nimbus juniper walrus harbor marble saffron lantern meadow thistle
cobalt ember willow puffin glacier tundra canyon oasis mosaic origami tango zephyr kiwi
papaya basil truffle biscuit muffin badger otter heron raven sparrow dune reef lagoon
""".split()
TLDS = ["com", "org", "net", "io", "dev", "ai", "app", "co", "co.za", "org.za", "gov", "edu",
        "info", "biz", "me", "tv", "uk", "co.uk", "de", "fr", "xyz", "site", "online", "tech"]
SUBS = ["www.", "www.", "blog.", "docs.", "news.", "shop.", "app.", "en.", "m.", "api.", "help."]
PATH_WORDS = ["about", "contact", "blog", "news", "products", "pricing", "docs", "wiki", "posts",
              "careers", "team", "faq", "store", "articles", "help", "guide", "results", "home", "list",
              "search", "category", "item", "page", "archive", "events", "jobs", "reviews", "menu"]
REAL_TRAIN_SITES = ["github.com", "news.ycombinator.com", "en.wikipedia.org", "bbc.com", "reddit.com",
                    "stackoverflow.com", "python.org", "nodejs.org", "nytimes.com", "imdb.com",
                    "amazon.com", "takealot.com", "news24.com", "chilledscore.com", "marvijo.com"]
REAL_TEST_SITES = ["example.com", "example.org", "iana.org", "quotes.toscrape.com", "books.toscrape.com",
                   "httpbin.org", "mozilla.org", "w3.org", "craigslist.org", "archive.org"]

VOWELS = "aeiou"
CONS = "bcdfghjklmnprstvwz"


def syllable_word(rng):
    return "".join(rng.choice(CONS) + rng.choice(VOWELS) for _ in range(rng.randint(2, 4)))


def make_name(rng, words):
    r = rng.random()
    if r < 0.35:
        return rng.choice(words)
    if r < 0.6:
        return rng.choice(words) + rng.choice(["", "-"]) + rng.choice(words)
    if r < 0.75:
        return rng.choice(words) + str(rng.randint(1, 999))
    if r < 0.9:
        return syllable_word(rng)
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(rng.randint(3, 10)))


def make_url(rng, words, real_sites):
    """Return (surface form typed by the user, canonical url the model must write)."""
    if rng.random() < 0.12:
        host = rng.choice(real_sites)
    else:
        host = make_name(rng, words) + "." + rng.choice(TLDS)
        if rng.random() < 0.3:
            host = rng.choice(SUBS) + host
    path = ""
    r = rng.random()
    if r < 0.25:
        path = "/" + rng.choice(PATH_WORDS)
    elif r < 0.35:
        path = "/" + rng.choice(PATH_WORDS) + "/" + rng.choice(words + PATH_WORDS)
    elif r < 0.42:
        path = "/" + rng.choice(PATH_WORDS) + "/" + str(rng.randint(1, 9999))
    elif r < 0.47:
        path = "/" + rng.choice(PATH_WORDS) + "?q=" + rng.choice(words)
    elif r < 0.5:
        path = "/" + rng.choice(words) + "-" + rng.choice(words)
    elif r < 0.56:
        # wiki style: /wiki/Some_Title_Here
        parts = [rng.choice(words).capitalize() for _ in range(rng.randint(1, 3))]
        path = "/" + rng.choice(["wiki", "w", "page", "p"]) + "/" + rng.choice(["_", "-"]).join(parts)
    elif r < 0.6:
        path = "/" + rng.choice(PATH_WORDS) + "/" + rng.choice(words) + rng.choice([".html", ".php", ".htm", ".aspx"])
    elif r < 0.63:
        path = "/" + make_name(rng, words) + "/" + make_name(rng, words) + "/" + make_name(rng, words)
    if path and rng.random() < 0.08:
        path += "/"
    elif not path and rng.random() < 0.05:
        path = "/"
    scheme_r = rng.random()
    if scheme_r < 0.55:
        typed_scheme, scheme = "", "https://"
    elif scheme_r < 0.9:
        typed_scheme = scheme = "https://"
    else:
        typed_scheme = scheme = "http://"
    canonical = scheme + host + path
    typed_host = host
    if rng.random() < 0.06:
        typed_host = host.upper()
    elif rng.random() < 0.06:
        typed_host = host.capitalize()
    typed_scheme_s = typed_scheme.upper() if (typed_scheme and rng.random() < 0.05) else typed_scheme
    surface = typed_scheme_s + typed_host + path
    if rng.random() < 0.05:
        q = rng.choice(['"', "'", "<", "("])
        surface = q + surface + {'"': '"', "'": "'", "<": ">", "(": ")"}[q]
    return surface, canonical


def field_phrase(rng, fields):
    words = [rng.choice(FIELD_WORDS[f]) for f in fields]
    if rng.random() < 0.6:
        words = ["the " + w if not w.startswith("the ") and rng.random() < 0.7 else w for w in words]
    if rng.random() < 0.15:
        words[0] = "all " + words[0].replace("the ", "")
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return words[0] + rng.choice([" and ", " & ", ", ", " plus "]) + words[1]
    return ", ".join(words[:-1]) + rng.choice([", and ", " and ", ", "]) + words[-1]


def pick_fields(rng):
    k = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
    fields = rng.sample(FIELDS, k)
    order = list(fields)
    rng.shuffle(order)  # the user may name them in any order
    canonical = [f for f in FIELDS if f in fields]
    return order, canonical


def typo(rng, s):
    if rng.random() > 0.08 or len(s) < 6:
        return s
    i = rng.randint(0, len(s) - 2)
    if s[i] == " " or s[i + 1] == " ":
        return s
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def protect_url(text, surface, fn):
    """Apply fn to text but leave the url untouched."""
    if not surface:
        return fn(text)
    head, _, tail = text.partition(surface)
    return fn(head) + surface + fn(tail)


def style(rng, text, surface):
    text = PREFIX[rng.randrange(len(PREFIX))] + text + SUFFIX[rng.randrange(len(SUFFIX))]
    r = rng.random()
    if r < 0.12:
        text = text[:1].upper() + text[1:]
    elif r < 0.16:
        text = protect_url(text, surface, str.upper)
    text = protect_url(text, surface, lambda s: " ".join(typo(rng, w) for w in s.split(" ")))
    if rng.random() < 0.04:
        text = text.replace(" ", "  ", 1)
    return text


def out_json(url, fields):
    return json.dumps({"url": url, "extract": fields}, separators=(",", ":"))


ERR_SCOPE = json.dumps({"error": "out_of_scope"}, separators=(",", ":"))
ERR_NOURL = json.dumps({"error": "no_url"}, separators=(",", ":"))


def fake_email(rng):
    return rng.choice(["bob", "anna", "thabo", "sales", "info", "me"]) + "@" + make_name(rng, TRAIN_WORDS) + "." + rng.choice(TLDS)


def make_example(rng, field_t, generic_t, words, real_sites, kind=None):
    kind = kind or rng.choices(["field", "generic", "nourl", "scope"], weights=[0.5, 0.3, 0.07, 0.13])[0]
    if kind == "scope":
        t = rng.choice(OUT_OF_SCOPE).replace("{e}", fake_email(rng))
        return style(rng, t, ""), ERR_SCOPE
    if kind == "nourl":
        order, _ = pick_fields(rng)
        t = rng.choice(NO_URL_T).replace("{f}", field_phrase(rng, order)).replace("{g}", rng.choice(GENERIC))
        return style(rng, t, ""), ERR_NOURL
    surface, canonical = make_url(rng, words, real_sites)
    if kind == "field":
        order, fields = pick_fields(rng)
        t = rng.choice(field_t).replace("{f}", field_phrase(rng, order))
    else:
        fields = ["all"]
        t = rng.choice(generic_t).replace("{g}", rng.choice(GENERIC))
    t = t.replace("{u}", surface)
    return style(rng, t, surface), out_json(canonical, fields)


MAX_IN = 150


def build(rng, n, field_t, generic_t, words, real_sites, kind=None):
    rows, seen = [], set()
    while len(rows) < n:
        x, y = make_example(rng, field_t, generic_t, words, real_sites, kind)
        if len(x) > MAX_IN or x in seen:
            continue
        seen.add(x)
        rows.append({"input": x, "output": y})
    return rows


def write(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--train", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    train = build(rng, a.train, TRAIN_FIELD_T, TRAIN_GENERIC_T, TRAIN_WORDS, REAL_TRAIN_SITES)
    val = build(random.Random(a.seed + 1), 2000, TRAIN_FIELD_T, TRAIN_GENERIC_T, TRAIN_WORDS, REAL_TRAIN_SITES)
    t_dom = build(random.Random(a.seed + 2), 1000, TRAIN_FIELD_T, TRAIN_GENERIC_T, TEST_WORDS, REAL_TEST_SITES)
    t_phr = build(random.Random(a.seed + 3), 1000, TEST_FIELD_T, TEST_GENERIC_T, TRAIN_WORDS, REAL_TRAIN_SITES)
    train_inputs = {r["input"] for r in train}
    val = [r for r in val if r["input"] not in train_inputs]
    # Proof that the demo domain is unseen.
    leaks = [r for r in train if "example" in (r["input"] + r["output"]).lower()]
    assert not leaks, f"'example' leaked into training data: {leaks[:3]}"
    write(f"{a.out}/train.jsonl", train)
    write(f"{a.out}/val.jsonl", val)
    write(f"{a.out}/test_unseen_domains.jsonl", t_dom)
    write(f"{a.out}/test_unseen_phrasings.jsonl", t_phr)
    stats = {"train": len(train), "val": len(val), "test_unseen_domains": len(t_dom),
             "test_unseen_phrasings": len(t_phr), "seed": a.seed,
             "max_input_chars": MAX_IN, "train_contains_example": False}
    with open(f"{a.out}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats))
    for r in train[:12]:
        print(r["input"], "=>", r["output"])


if __name__ == "__main__":
    main()
