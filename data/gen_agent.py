"""Synthetic tool-calling episodes for MarvijoWebSearch v4 (the agent).

v3 turned a request into ONE plan and only worked when the request held a url.
v4 is an agent: it writes one Puppeteer tool call, reads what the tool returned, then
writes the next call, until it has the data or runs out of options.

One training row is one whole episode, held in one character sequence:

    <bos>request<sep>call 1<eos>result 1<sep>call 2<eos>result 2<sep> ... call N<eos>

Loss is computed only on the calls (and their <eos>), never on the request or on a tool
result (AgentTuning 2023 does the same). Tool results here use exactly the strings that
runtime/tools.js prints, so training and the live runtime look the same to the model.

The calls come from an expert policy (expert_step below) run against a simulated web:
pages time out, 404, block bots, have no images, or do not mention today's date. About a
third of episodes hold at least one failure, so the model learns to retry or switch source
(AgentRefine 2025 and NAT 2024 show failure traces help).

Request kinds:
- url      : "scrape the links from news24.com"            -> goto that url
- recipe   : "image links from wikipedia for Cape_Town"    -> goto a url the model builds
- search   : "today's uefa fixtures"                       -> search, then open a result
- no_target: "extract the headings"                        -> fail no_target
- scope    : "write me a poem"                             -> fail out_of_scope

Usage: python gen_agent.py --out build_v4 --train 120000
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_data as v3  # noqa: E402  (v3 url requests, phrasings, styling)

# v4 adds a few field words people used in the brief ("image links from wikipedia ...").
v3.FIELD_WORDS["images"] += ["image links", "image link", "img links", "picture links", "photo urls"]
v3.FIELD_WORDS["links"] += ["all the links", "link addresses"]

MAX_SEQ = 768
COPY_RATE = 0.0  # --copy: share of training names that are invented, so the model learns to copy words
TYPO_RATE = 0.0  # --typo: share of training requests whose real topic name gets a forced typo (gold stays correct)
MIX_RATE = 0.0   # --mix: share of training search requests phrased with generated filler (see mix_template)


def call(tool, **kw):
    return json.dumps({"tool": tool, **kw}, separators=(",", ":"), ensure_ascii=True)


# ------------------------------------------------------------------ vocabulary
# Each list is split: words before the "|" are for training, words after it are only in
# the unseen-topics test, so that test measures copying words the model never saw.
def split(s):
    a, b = s.split("|")
    return a.split(","), b.split(",")


LEAGUES = split("uefa,premier league,la liga,serie a,bundesliga,ligue 1,champions league,europa league,psl,"
                "mls,nba,nfl,ipl,f1,ufc,wimbledon,afcon,carabao cup,fa cup,super rugby,urc,nhl,mlb,"
                "conference league,eredivisie,copa libertadores,six nations"
                "|liga portugal,scottish premiership,a-league,big bash,euroleague,dfb pokal,coppa italia,"
                "currie cup,j1 league,saudi pro league")
SPORT_THINGS = split("fixtures,results,standings,scores,table,log,schedule,top scorers,match results,"
                     "live scores,news,games,transfers|kickoff times,line-ups,match report")
CITIES = split("cape town,johannesburg,durban,pretoria,london,paris,new york,tokyo,nairobi,lagos,berlin,"
               "madrid,rome,sydney,toronto,dubai,cairo,accra,lisbon,amsterdam,mumbai,seoul,chicago,"
               "boston,soweto,stellenbosch,gqeberha,bloemfontein,kigali,harare"
               "|polokwane,windhoek,gaborone,oslo,lima,hanoi,perth,denver")
CITY_THINGS = split("weather in {c},{c} weather,population of {c},restaurants in {c},events in {c},"
                    "{c} news,things to do in {c},hotels in {c},{c} traffic,load shedding in {c},"
                    "{c} public holidays,jobs in {c}|museums in {c},{c} sunrise time")
ASSETS = split("bitcoin,ethereum,gold,oil,usd to zar,eur to zar,gbp to zar,nvidia stock,apple stock,"
               "tesla stock,naspers share,jse top 40,petrol,s&p 500,nasdaq|solana,platinum,dogecoin,"
               "amazon stock")
ASSET_THINGS = split("{a} price,{a} rate,{a} exchange rate,price of {a},{a} chart,{a} news|{a} forecast")
TOPICS = split("ai,openai,nvidia,climate change,elections,python,javascript,rust,space x,nasa,tesla,"
               "apple,google,microsoft,eskom,springboks,bafana bafana,proteas,crypto,stock market,"
               "load shedding,taylor swift,the oscars,gpt,llama models,linux,react,docker,kubernetes"
               "|quantum computing,the olympics,mars rover,deepseek,chess,anthropic")
TOPIC_THINGS = split("{t} news,latest {t} news,{t} headlines,{t} updates,news about {t},{t} articles,"
                     "{t} release notes,{t} reviews|{t} blog posts,{t} rumours")
THINGS = split("iphone 17,ps5,macbook air,air fryer,running shoes,gaming laptop,electric car,rtx 5090,"
               "kindle,solar panels,coffee machine,4k tv,drone,smart watch,office chair"
               "|e-bike,vr headset,robot vacuum")
THING_THINGS = split("{x} price,{x} prices,{x} reviews,{x} specs,best {x},cheap {x},{x} deals,"
                     "{x} comparison|{x} discounts,{x} ratings")

TIME_WORDS = {  # surface word -> (word in the search query, date the tool filters on)
    "today's": ("today", "today"), "today": ("today", "today"), "tonight's": ("today", "today"),
    "tonight": ("today", "today"), "tomorrow's": ("tomorrow", "tomorrow"), "tomorrow": ("tomorrow", "tomorrow"),
    "yesterday's": ("yesterday", "yesterday"), "yesterday": ("yesterday", "yesterday"),
}
LOOSE_TIME = ["latest", "current", "this week's", "recent", "live", "new"]

# Search request templates: {q} is the topic phrase, {tw} a time word placed before it,
# {ta} a time word placed after it, {f} a field phrase.
SEARCH_T = split(
    "{tw} {q},{q} {ta},what are {tw} {q},what are the {q} {ta},give me {tw} {q},get me {tw} {q},"
    "find {tw} {q},show me {tw} {q},i want {tw} {q},look up {tw} {q},search for {tw} {q},"
    "can you find {tw} {q},check {tw} {q},fetch {tw} {q},what's {tw} {q},whats the {q} {ta},"
    "pull {tw} {q},grab {tw} {q} for me,go online and get {tw} {q},browse the web for {tw} {q},"
    "need {tw} {q},{q} {ta} please,{q} {ta}?,tell me {tw} {q},any {q} {ta},search the web for {q} {ta},"
    "use puppeteer to find {tw} {q},find out {tw} {q},google {tw} {q},look for {q} {ta},"
    "get {tw} {q} from the internet,i need the {q} {ta}"
    "|dig up {tw} {q},hunt down {tw} {q},i'm curious about {tw} {q},bring me {tw} {q},"
    "report {tw} {q},scout the web for {tw} {q}")
SEARCH_FIELD_T = split(
    "{f} of {q},{f} for {q},{f} about {q},find {f} of {q},get {f} for {q},search {q} and get {f},"
    "look up {q} and give me {f},{q} {f},i want {f} of {q},grab {f} for {q} from the web"
    "|harvest {f} about {q},collect {f} on {q}")

# Recipe sites: the model must build the url itself.
WIKI_T = split(
    "{f} from wikipedia for {x},{x} on wikipedia,wikipedia page for {x},wikipedia {x},{x} wiki page,"
    "get the wikipedia article on {x},{f} from the {x} wikipedia page,open wikipedia for {x} and get {f},"
    "{f} of the wikipedia article about {x},look up {x} on wikipedia,wiki {x},{f} on wikipedia for {x},"
    "go to wikipedia, {x}, give me {f},search wikipedia for {x}"
    "|wikipedia's {x} entry,{f} off wikipedia about {x}")
WIKI_TOPICS = split("cape town,table mountain,nelson mandela,python (programming language),machine learning,"
                    "eiffel tower,albert einstein,south africa,world war ii,the beatles,photosynthesis,"
                    "black hole,ancient egypt,tokyo,mount kilimanjaro,durban,kruger national park,linux,"
                    "transformer (deep learning architecture),fifa world cup,lionel messi,soweto,"
                    "johannesburg,robben island,great wall of china,moon,jupiter,javascript,bitcoin,"
                    "large language model,shaka,zulu people,isiXhosa,springbok"
                    "|kalahari,okavango delta,ada lovelace,alan turing,mapungubwe,drakensberg")
GITHUB_T = split("github repo {x},{x} on github,{f} from the github repo {x},open github {x},"
                 "github {x} readme,the {x} repository on github|{x}'s github page")
GITHUB_REPOS = split("torvalds/linux,karpathy/nanogpt,facebook/react,microsoft/vscode,python/cpython,"
                     "puppeteer/puppeteer,nodejs/node,openai/whisper,huggingface/transformers,"
                     "denoland/deno,vercel/next.js,rust-lang/rust,golang/go,ggml-org/llama.cpp,"
                     "pytorch/pytorch,tensorflow/tensorflow,microsoft/typescript,sveltejs/svelte"
                     "|anthropics/anthropic-sdk-python,karpathy/llm.c,ollama/ollama")
REDDIT_T = split("r/{x},{f} from r/{x},the {x} subreddit,top posts on r/{x},reddit {x},what's new on r/{x},"
                 "{f} on the {x} subreddit|latest from r/{x}")
SUBREDDITS = split("southafrica,programming,localllama,soccer,worldnews,python,javascript,machinelearning,"
                   "chess,capetown,johannesburg,technology,science,gaming,space,webdev|rust,formula1,dataisbeautiful")
PKG_T = split("npm package {x},{x} on npm,the npm page for {x},pypi package {x},{x} on pypi,"
              "the pypi page for {x}|{x} npm docs,{x} pypi release")
NPM_PKGS = split("react,express,puppeteer,lodash,axios,vite,zod,typescript,next,chalk,commander,dotenv,"
                 "playwright,prettier,eslint|hono,drizzle-orm")
PYPI_PKGS = split("requests,numpy,pandas,torch,flask,django,fastapi,pydantic,scrapy,beautifulsoup4,"
                  "httpx,pytest,transformers,openai|polars,uv")
HN_T = split("hacker news front page,top stories on hacker news,{f} from hacker news,whats on hacker news,"
             "hn front page,hacker news today,latest on hacker news|hacker news headlines")
BBC_T = split("bbc news headlines,latest bbc news,{f} from bbc news,bbc news today,"
              "what's on bbc news|bbc news front page")

# v4 out of scope: actions, not reading the web. (v3 also refused questions like "what's the
# weather today"; v4 answers those by searching.)
OUT_OF_SCOPE = split(
    "write me a poem about cats,tell me a joke,translate hello into french,write a python function to sort a list,"
    "summarize this paragraph for me,send an email to {e},email {e} the report,book a flight to cape town,"
    "set a timer for 5 minutes,play some music,write a cover letter,help me with my homework,thanks,"
    "turn off the lights,order a pizza,what's your name,delete my files,reboot the server,call mom,"
    "remind me to buy milk,draw a picture of a dog,sing a song,fix this bug in my code,generate a password,"
    "who are you,log into my bank account,buy bitcoin,ok,yes,no,write a story about a dragon,"
    "message {e} that i am late,make me a todo list,can you hear me,do my taxes,rate my outfit,count to ten,"
    "hello,hi there,how are you,post this on twitter,pay my electricity bill,schedule a meeting for 3pm,"
    "rename my files,install chrome,open spotify,text {e} happy birthday,transfer r500 to bob"
    "|compose a haiku,cancel my subscription,tidy up my desktop,book me a haircut")
NO_TARGET_T = [
    "scrape the website and give me {f}", "go to the site and get {g}", "extract {f} from the page",
    "open the website and pull {g}", "scrape it", "get {f} from that site", "go to the link i sent and get {g}",
    "visit the homepage and grab {f}", "fetch the page", "extract {g}", "scrape {f} from my site",
    "go to the url and give me {g}", "crawl the website", "get {g} from the web page", "get me {f} from the page",
    "pull {g} from the article i sent earlier", "get {f}", "extract {f} please", "{f} from the page",
    "get {g} off the site we talked about", "scrape this", "open it and get {f}",
]

SITE_WORDS = ["news", "sport", "live", "info", "daily", "hub", "central", "today", "zone", "times", "wire", "now",
              "online", "guide", "report", "insider", "world", "media", "post", "score", "pulse", "watch"]
RESULT_PATHS = ["", "news", "results", "fixtures", "scores", "live", "latest", "article", "{y}/{m}/{s}", "{s}",
                "en/{s}", "sport/{s}", "{s}-{n}", "story/{n}", "p/{n}", "wiki/{S}", "category/{s}",
                "{s}/{s2}", "tag/{s}", "item?id={n}", "{s}.html", "markets/{s}"]


# ------------------------------------------------------------------ helpers
def pick(rng, lists, test, shape=None):
    """A word from the train or test half of a list. With --copy, training rows sometimes get an
    invented name of the given shape instead, so the model cannot memorize the list and must copy."""
    if COPY_RATE and shape and not test and rng.random() < COPY_RATE:
        INVENTED.append(invented(rng, shape))
        return INVENTED[-1]
    return rng.choice(lists[1] if test else lists[0])


# --mix filler: openers, verbs, objects, prepositions and trailers combined at random around the topic,
# so the model learns where a topic starts and ends instead of copying everything after a known opener.
# None of these words appear only in test templates (checked in main).
MIX_OPENERS = ["", "", "", "could you ", "would you ", "can u ", "i'd like you to ", "i need you to ", "kindly ",
               "go ahead and ", "real quick, ", "when you can, ", "be a star and ", "if possible ", "ok so ",
               "right, ", "so, ", "hmm, ", "listen, ", "help me ", "try to ", "i want you to ", "just ",
               "my friend asked me to ", "for tomorrow's meeting, ", "boss wants me to "]
MIX_VERBS = ["fetch", "grab", "pull", "get", "find", "snag", "round up", "chase up", "track", "gather", "pick up",
             "retrieve", "source", "compile", "nab", "scoop up", "rustle up", "locate", "check", "show", "list",
             "send me", "hand me", "drop me", "dish up", "pull up", "look into", "research", "pull together",
             "get hold of", "dredge up", "go and get", "fish out", "sniff out", "search out", "turn up", "unearth",
             "serve up", "look up", "find out", "give me", "get me", "show me", "tell me"]
MIX_OBJECTS = ["", "", "", "", "me ", "us ", "info on ", "anything about ", "stuff about ", "details on ",
               "a rundown of ", "the lowdown on ", "whatever you can on ", "some info about ", "what's new with ",
               "everything on ", "the facts on ", "a summary of ", "what you can find on "]
MIX_PREPS = ["about", "on", "for", "of", "from", "regarding", "concerning", "related to", "covering", "around",
             "on the topic of", "to do with", "in", "behind"]
MIX_TRAILERS = ["", "", "", "", " for my essay", " for a project", " for my notes", " for the team", " thanks a lot",
                " cheers", " when you get a chance", " if you can", " no rush", " for my boss", " for later",
                " for my presentation", " please and thanks", " ty", " much appreciated", " for a school assignment",
                " real quick", " on the web", " online", " from the internet", " using the browser", " via google"]


def mix_template(rng, field):
    o, v = rng.choice(MIX_OPENERS), rng.choice(MIX_VERBS)
    if field:
        body = f"{v} {{f}} {rng.choice(MIX_PREPS)} {{q}}"
    elif rng.random() < 0.5:
        body = f"{v} {rng.choice(MIX_OBJECTS)}{{tw}} {{q}}"
    else:
        body = f"{v} {rng.choice(MIX_OBJECTS)}{{q}} {{ta}}"
    return o + body + rng.choice(MIX_TRAILERS)


def force_typo(rng, phrase):
    """Swap two neighbouring letters in one word of 4+ letters (the kind of typo v3.typo makes)."""
    words = phrase.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not idx:
        return phrase
    i = rng.choice(idx)
    w = words[i]
    j = rng.randint(0, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


INVENTED = []  # names invented for the current request: typos must not touch them (no way to undo one)
TEST_ONLY = set()  # every test-half word (filled in below), never invented for training


def fake_word(rng):
    r = rng.random()
    if r < 0.5:
        w = v3.syllable_word(rng)
    elif r < 0.7:
        w = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 9)))
    elif r < 0.85:
        w = v3.make_name(rng, v3.TRAIN_WORDS)
    else:
        w = v3.syllable_word(rng) + str(rng.randint(1, 99))
    return w.lower()


def invented(rng, shape):
    while True:
        if shape == "words":  # a topic, league, city or product: 1 to 3 words
            w = " ".join(fake_word(rng) for _ in range(rng.choices([1, 2, 3], weights=[5, 4, 1])[0]))
            if rng.random() < 0.1:
                w = w.replace(" ", "-")
        elif shape == "repo":  # owner/name
            w = fake_word(rng) + "/" + rng.choice(["-", "_", ".", ""]).join(fake_word(rng) for _ in range(rng.randint(1, 3)))
        elif shape == "pkg":
            w = "-".join(fake_word(rng) for _ in range(rng.choice([1, 1, 2])))
        else:  # "slug": a subreddit
            w = "".join(fake_word(rng) for _ in range(rng.choice([1, 1, 2])))
        if w not in TEST_ONLY and "example" not in w:
            return w


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def wiki_title(topic):
    t = topic.strip().replace(" ", "_")
    return t[:1].upper() + t[1:]


def result_url(rng, q):
    """A plausible search-result url for query q."""
    words = [w for w in re.split(r"[^a-z0-9]+", q.lower()) if w]
    base = rng.choice([slug(rng.choice(words)) if words else "site", v3.make_name(rng, v3.TRAIN_WORDS)])
    host = base + rng.choice(["", "", rng.choice(SITE_WORDS)])
    sub = rng.choice(["www.", "www.", "", "", "en.", "m.", "news.", "sport."])
    tld = rng.choice(v3.TLDS + v3.TLDS2)
    s = slug("-".join(rng.sample(words, min(len(words), rng.randint(1, 3))))) or "page"
    path = rng.choice(RESULT_PATHS).format(s=s, s2=slug(rng.choice(v3.PATH_WORDS)), n=rng.randint(10, 999999),
                                           y=rng.randint(2019, 2026), m=f"{rng.randint(1, 12):02d}", S=s.title())
    url = f"https://{sub}{host}.{tld}/{path}"
    if rng.random() < 0.3 and not path.endswith("/") and path:
        url += "/"
    return url[:90]


def counts_obs(rng, need):
    """An 'ok' observation. need = fields that must be non-zero (None = any page)."""
    c = {"text": rng.randint(1, 900), "links": rng.randint(0, 2500), "images": rng.randint(0, 200),
         "tables": rng.choice([0, 0, 0, 1, 2, rng.randint(0, 60)])}
    for k in c:
        if rng.random() < 0.12:
            c[k] = 0
    for k in need or []:
        if k in c and c[k] == 0:
            c[k] = rng.randint(1, 300)
    status = rng.choice([200] * 12 + [203, 301])
    if status == 301:
        status = 200
    return f"ok {status} " + " ".join(f"{k}:{c[k]}" for k in ["text", "links", "images", "tables"]), c


COUNTED = ["text", "links", "images", "tables"]


def needed(fields):
    """Fields the page must have for the request to make sense (counted ones only)."""
    if fields == ["all"]:
        return ["text"]
    return [f for f in fields if f in COUNTED]


# ------------------------------------------------------------------ requests
def make_request(rng, test_words, test_t, kind=None):
    """Returns dict(text, kind, target..., fields, find, q)."""
    INVENTED.clear()
    kind = kind or rng.choices(["url", "recipe", "search", "no_target", "scope"],
                               weights=[0.30, 0.22, 0.36, 0.05, 0.07])[0]
    tw = test_words
    if kind == "url":
        x, y = v3.make_example(rng, v3.TEST_FIELD_T if test_t else v3.TRAIN_FIELD_T,
                               v3.TEST_GENERIC_T if test_t else v3.TRAIN_GENERIC_T,
                               v3.TEST_WORDS if tw else v3.TRAIN_WORDS,
                               v3.REAL_TEST_SITES if tw else v3.REAL_TRAIN_SITES,
                               kind=rng.choice(["field", "field", "generic"]))
        plan = json.loads(y)
        host = re.sub(r"^https?://(www\.)?", "", plan["url"]).split("/")[0]
        return {"text": x, "kind": "url", "url": plan["url"], "fields": plan["extract"], "find": None,
                "q": host}
    if kind == "scope":
        t = pick(rng, OUT_OF_SCOPE, test_t).replace("{e}", v3.fake_email(rng))
        if rng.random() < 0.2:  # a url does not make an action in scope
            surface, _ = v3.make_url(rng, v3.TRAIN_WORDS, v3.REAL_TRAIN_SITES)
            t = rng.choice(v3.SCOPE_URL_T).replace("{e}", v3.fake_email(rng)).replace("{u}", surface)
            return {"text": v3.style(rng, t, surface), "kind": "scope"}
        return {"text": v3.style(rng, t, ""), "kind": "scope"}
    if kind == "no_target":
        order, _ = v3.pick_fields(rng)
        t = rng.choice(NO_TARGET_T).replace("{f}", v3.field_phrase(rng, order)).replace("{g}", rng.choice(v3.GENERIC))
        return {"text": v3.style(rng, t, ""), "kind": "no_target"}

    # fields named in the request (else text, or all for a plain page)
    if rng.random() < 0.35:
        order, fields = v3.pick_fields(rng)
        fphrase = v3.field_phrase(rng, order)
    else:
        fields, fphrase = None, None

    if kind == "recipe":
        site = rng.choices(["wiki", "github", "reddit", "npm", "pypi", "hn", "bbc"],
                           weights=[0.4, 0.13, 0.13, 0.08, 0.08, 0.09, 0.09])[0]
        if site == "wiki":
            topic = pick(rng, WIKI_TOPICS, tw, "words")
            if rng.random() < 0.3:
                topic = wiki_title(topic)  # people also type Cape_Town
            elif rng.random() < 0.3:
                topic = topic.title()
            tpl, url = pick(rng, WIKI_T, test_t), f"https://en.wikipedia.org/wiki/{wiki_title(topic)}"
            q = topic.replace("_", " ") + " wikipedia"
        elif site == "github":
            topic = pick(rng, GITHUB_REPOS, tw, "repo")
            tpl, url, q = pick(rng, GITHUB_T, test_t), f"https://github.com/{topic}", topic + " github"
        elif site == "reddit":
            topic = pick(rng, SUBREDDITS, tw, "slug")
            tpl, url, q = pick(rng, REDDIT_T, test_t), f"https://old.reddit.com/r/{topic}/", "r/" + topic
        elif site in ("npm", "pypi"):
            tpl = rng.choice([t for t in (PKG_T[1] if test_t else PKG_T[0]) if site in t])
            topic = pick(rng, NPM_PKGS if site == "npm" else PYPI_PKGS, tw, "pkg")
            url = (f"https://www.npmjs.com/package/{topic}" if site == "npm" else f"https://pypi.org/project/{topic}/")
            q = f"{topic} {site}"
        elif site == "hn":
            topic, tpl, url, q = "", pick(rng, HN_T, test_t), "https://news.ycombinator.com/", "hacker news"
        else:
            topic, tpl, url, q = "", pick(rng, BBC_T, test_t), "https://www.bbc.com/news", "bbc news"
        if "{f}" in tpl and not fphrase:
            fields = rng.choice([["images"], ["links"], ["text"], ["headings"], ["tables"], ["images", "links"]])
            fphrase = v3.field_phrase(rng, fields)
        if "{f}" not in tpl:
            fields = None
        text = tpl.replace("{x}", topic).replace("{f}", fphrase or "")
        fields = fields or (["all"] if site in ("wiki", "github", "npm", "pypi") else ["links"] if site == "hn" else ["text"])
        keep = topic if INVENTED and topic and topic in text else ""
        if TYPO_RATE and not INVENTED and not tw and topic and topic in text and rng.random() < TYPO_RATE:
            text = text.replace(topic, force_typo(rng, topic), 1)  # the url keeps the right spelling
        return {"text": v3.style(rng, text, keep), "kind": "recipe", "url": url, "fields": fields, "find": None, "q": q}

    # kind == "search"
    group = rng.choices(["sport", "city", "asset", "topic", "thing"], weights=[0.34, 0.18, 0.14, 0.2, 0.14])[0]
    if group == "sport":
        q = pick(rng, LEAGUES, tw, "words") + " " + pick(rng, SPORT_THINGS, tw, "words")
    elif group == "city":
        q = pick(rng, CITY_THINGS, tw).replace("{c}", pick(rng, CITIES, tw, "words"))
    elif group == "asset":
        q = pick(rng, ASSET_THINGS, tw).replace("{a}", pick(rng, ASSETS, tw, "words"))
    elif group == "topic":
        q = pick(rng, TOPIC_THINGS, tw).replace("{t}", pick(rng, TOPICS, tw, "words"))
    else:
        q = pick(rng, THING_THINGS, tw).replace("{x}", pick(rng, THINGS, tw, "words"))
    find = None
    if fields:
        if MIX_RATE and not test_t and rng.random() < MIX_RATE:
            tpl = mix_template(rng, True)
        else:
            tpl = pick(rng, SEARCH_FIELD_T, test_t)
        text = tpl.replace("{f}", fphrase).replace("{q}", q)
        query = q
    else:
        fields = ["text"]
        if MIX_RATE and not test_t and rng.random() < MIX_RATE:
            tpl = mix_template(rng, False)
        else:
            tpl = pick(rng, SEARCH_T, test_t)
        r = rng.random()
        if r < 0.45 and group in ("sport", "city", "asset", "topic"):
            surface = rng.choice(list(TIME_WORDS))
            qword, find = TIME_WORDS[surface]
            if "{ta}" in tpl:
                surface = qword  # "results today", not "results today's"
        elif r < 0.6:
            surface, qword = rng.choice(LOOSE_TIME), None
            if "{ta}" in tpl:
                surface = rng.choice(["latest", "now", "right now"])
        else:
            surface, qword = "", None
        text = tpl.replace("{tw}", surface if "{tw}" in tpl else "").replace("{ta}", surface if "{ta}" in tpl else "")
        text = re.sub(r"\s+", " ", text.replace("{q}", q)).strip()
        text = text.replace("the today", "today").replace(" ?", "?")
        query = q + (" " + qword if qword else "")
    keep = q if INVENTED and q in text else ""
    if TYPO_RATE and not INVENTED and not tw and q in text and rng.random() < TYPO_RATE:
        text = text.replace(q, force_typo(rng, q), 1)  # the query keeps the right spelling
    return {"text": v3.style(rng, text, keep), "kind": "search", "fields": fields, "find": find, "q": query}


# ------------------------------------------------------------------ expert policy + simulated web
def extract_call(req):
    kw = {"fields": req["fields"]}
    if req.get("find"):
        kw["find"] = req["find"]
    return call("extract", **kw)


def episode(rng, req):
    """Run the expert policy against a simulated web. Returns list of (call, observation)."""
    if req["kind"] == "scope":
        return [(call("fail", reason="out_of_scope"), None)]
    if req["kind"] == "no_target":
        return [(call("fail", reason="no_target"), None)]

    steps = []
    queue = []            # untried urls from the last search
    searched = False
    tried = set()
    timeouts = set()
    first = req.get("url")
    fail_rate = rng.choice([0.0, 0.0, 0.15, 0.3, 0.45])  # some episodes run on a flaky web

    if first:
        url = first
        act = call("goto", url=url)
    else:
        act, url = call("search", q=req["q"]), None
    while True:
        if len(steps) >= 9:
            steps.append((call("fail", reason="not_found"), None))
            return steps
        tool = json.loads(act)["tool"]
        if tool == "search":
            searched = True
            r = rng.random()
            n = 0 if r < 0.03 else rng.choice([1, 2, 3, 3, 3, 3])
            urls = []
            for _ in range(n):
                u = result_url(rng, req["q"])
                if u not in urls:
                    urls.append(u)
            if req["kind"] == "recipe" and urls and rng.random() < 0.5:
                urls[0] = req["url"] if req["url"] not in tried else urls[0]
            obs = "results " + " ".join(f"{i + 1} {u}" for i, u in enumerate(urls)) if urls else "results none"
            steps.append((act, obs))
            queue = [u for u in urls if u not in tried]
            if not queue:
                steps.append((call("fail", reason="not_found"), None))
                return steps
            url = queue.pop(0)
            act = call("goto", url=url)
            continue
        if tool == "goto":
            tried.add(url)
            r = rng.random()
            if r < fail_rate:
                err = rng.choices(["timeout", "404", "blocked", "dns", "500", "unreachable", "zero"],
                                  weights=[3, 3, 3, 1, 1, 1, 3])[0]
            else:
                err = None
            if err == "timeout":
                steps.append((act, "error timeout"))
                if url not in timeouts:
                    timeouts.add(url)
                    continue  # same call again: a timeout is often a one-off
            elif err in ("404", "blocked", "dns", "500", "unreachable"):
                steps.append((act, f"error {err}"))
            else:
                obs, c = counts_obs(rng, needed(req["fields"]))
                if err == "zero":  # the page loads but lacks what we need
                    for k in needed(req["fields"]) or ["text"]:
                        obs = re.sub(rf"{k}:\d+", f"{k}:0", obs)
                    steps.append((act, obs))
                else:
                    steps.append((act, obs))
                    # extract
                    act = extract_call(req)
                    want = req["fields"]
                    fields = ["title", "meta", "headings", "text", "links", "images", "tables"] if want == ["all"] else want
                    if req.get("find") and rng.random() < 0.3:
                        got = "got text:0"
                    else:
                        parts = []
                        for f in fields:
                            if f in COUNTED:
                                parts.append(f"{f}:{c[f] if not req.get('find') else rng.randint(1, 80)}")
                            elif f == "headings":
                                parts.append(f"headings:{rng.randint(1, 60)}")
                        got = "got " + (" ".join(parts) if parts else " ".join(fields))
                    steps.append((act, got))
                    if got != "got text:0":
                        steps.append((call("done"), None))
                        return steps
            # this url failed: next result, else search, else give up
            queue = [u for u in queue if u not in tried]
            if queue:
                url = queue.pop(0)
                act = call("goto", url=url)
            elif not searched:
                act, url = call("search", q=req["q"]), None
            else:
                steps.append((call("fail", reason="not_found"), None))
                return steps


def render(req, steps):
    """The text sequence (for stats and debugging); train.py builds token ids from rows."""
    s = req["text"] + " <sep> "
    for act, obs in steps:
        s += act + " <eos> " + (obs + " <sep> " if obs is not None else "")
    return s


def seq_len(req, steps):
    n = 2 + len(req["text"])
    for act, obs in steps:
        n += len(act) + 1 + (len(obs) + 1 if obs is not None else 0)
    return n


def build(rng, n, test_words=False, test_t=False, kind=None, avoid=frozenset()):
    rows, seen = [], set()
    while len(rows) < n:
        req = make_request(rng, test_words, test_t, kind)
        if len(req["text"]) > 220 or req["text"] in seen or req["text"] in avoid:
            continue
        steps = episode(rng, req)
        if seq_len(req, steps) > MAX_SEQ:
            continue
        seen.add(req["text"])
        rows.append({"input": req["text"], "kind": req["kind"],
                     "steps": [{"call": a, "obs": o} for a, o in steps]})
    return rows


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build_v4")
    ap.add_argument("--train", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--copy", type=float, default=0.0, help="share of training names that are invented")
    ap.add_argument("--mix", type=float, default=0.0, help="share of training searches phrased with --mix filler")
    ap.add_argument("--typo", type=float, default=0.0, help="share of training requests with a forced typo in a real name")
    a = ap.parse_args()
    global COPY_RATE, MIX_RATE, TYPO_RATE
    COPY_RATE, MIX_RATE, TYPO_RATE = a.copy, a.mix, a.typo
    test_tpl_words = set()
    train_tpl_words = set()
    for lists in (SEARCH_T, SEARCH_FIELD_T, WIKI_T, GITHUB_T, REDDIT_T, PKG_T, HN_T, BBC_T):
        train_tpl_words.update(w for t in lists[0] for w in re.findall(r"[a-z']+", t))
        test_tpl_words.update(w for t in lists[1] for w in re.findall(r"[a-z']+", t))
    mix_words = {w for part in (MIX_OPENERS, MIX_VERBS, MIX_OBJECTS, MIX_PREPS, MIX_TRAILERS)
                 for t in part for w in re.findall(r"[a-z']+", t)}
    leak = mix_words & (test_tpl_words - train_tpl_words)
    assert not leak, f"--mix filler uses test-only template words: {leak}"
    for lists in (LEAGUES, SPORT_THINGS, CITIES, ASSETS, TOPICS, THINGS, WIKI_TOPICS, GITHUB_REPOS, SUBREDDITS,
                  NPM_PKGS, PYPI_PKGS):
        TEST_ONLY.update(lists[1])
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    train = build(rng, a.train)
    COPY_RATE = MIX_RATE = TYPO_RATE = 0.0  # val and test sets stay plain, so runs with and without --copy score on the same rows
    tin = {r["input"] for r in train}
    val = build(random.Random(a.seed + 1), 1500, avoid=tin)
    t_topics = build(random.Random(a.seed + 2), 600, test_words=True, avoid=tin)
    t_phr = build(random.Random(a.seed + 3), 600, test_t=True, avoid=tin)
    leaks = [r for r in train if "example" in json.dumps(r).lower()]
    assert not leaks, f"'example' leaked into training data: {leaks[:2]}"
    write(f"{a.out}/train.jsonl", train)
    write(f"{a.out}/val.jsonl", val)
    write(f"{a.out}/test_unseen_topics.jsonl", t_topics)
    write(f"{a.out}/test_unseen_phrasings.jsonl", t_phr)
    lens = sorted(seq_len({"text": r["input"]}, [(s["call"], s["obs"]) for s in r["steps"]]) for r in train)
    kinds = {}
    for r in train:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    def has_failure(r):  # a tool error, an empty result, or more than one page opened
        obs = [s["obs"] or "" for s in r["steps"]]
        gotos = sum('"goto"' in s["call"] for s in r["steps"])
        return gotos > 1 or any(o.startswith("error") or o in ("results none", "got text:0") for o in obs)
    with_fail = sum(has_failure(r) for r in train)
    stats = {"train": len(train), "val": len(val), "test_unseen_topics": len(t_topics),
             "test_unseen_phrasings": len(t_phr), "seed": a.seed, "copy": a.copy, "mix": a.mix, "typo": a.typo, "kinds": kinds,
             "episodes_with_a_failure": round(with_fail / len(train), 3),
             "seq_len_median": lens[len(lens) // 2], "seq_len_p99": lens[int(len(lens) * 0.99)],
             "seq_len_max": lens[-1], "calls_per_episode": round(sum(len(r["steps"]) for r in train) / len(train), 2)}
    with open(f"{a.out}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=1))
    for r in train[:8]:
        print(render({"text": r["input"]}, [(s["call"], s["obs"]) for s in r["steps"]]))


if __name__ == "__main__":
    main()
