"""Deterministic, country-agnostic text normalisation.

No external lookups: all maps below are hand-written string rules. `anyascii` (ISC
licence) transliterates any script (Kannada, Devanagari, Bengali, accents...) to ASCII.
"""
import re

from anyascii import anyascii

# ----------------------------------------------------------------- names
DBA_RE = re.compile(r"^.*?\b(?:trading as|doing business as|d/b/a|dba|t/a|aka|a\.k\.a\.?)\s+", re.I)
DOMAIN_RE = re.compile(r"\.(?:com|in|net|org|co|fr|us|io|biz|info)\b", re.I)
TLD_RE = re.compile(r"(?:www\.)?([a-z0-9-]+)\.(?:com|in|net|org|co|fr|us|io|biz|info)\b")
NOISE_RE = re.compile(r"\b(?:services?|servic|centers?|centres?)\b|\d{7,}")
HON_RE = re.compile(r"^(?:mr|mrs|ms|dr|messrs|m s)\s+")
SINGLE_RUN_RE = re.compile(r"\b[a-z](?: [a-z]\b)+")
LEGAL = frozenset((
    "private pvt limited ltd llc llp pllc plc lp inc incorporated corporation corp co company pc "
    "sarl sas sasu sa eurl sci snc gmbh "
    # transliterated legal suffixes (from native-script names via anyascii)
    "praivet praivt privet prayvet limitad limitd lmtd limiteda elelpi pra li"
).split())

# phonetic merges applied before vowel removal (order matters)
SUBS = [("ph", "f"), ("th", "t"), ("sh", "s"), ("ch", "k"), ("ck", "k"), ("c", "k"),
        ("q", "k"), ("z", "s"), ("w", "v"), ("x", "ks"), ("j", "g"), ("m", "n"), ("h", "")]


def norm_name(s: str) -> str:
    s = DBA_RE.sub("", s or "")
    s = anyascii(s).lower()
    s = TLD_RE.sub(r"\1", s)
    s = NOISE_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = HON_RE.sub("", s)
    s = SINGLE_RUN_RE.sub(lambda m: m.group(0).replace(" ", ""), s)   # "l l c" -> "llc"
    return " ".join(t for t in s.split() if t not in LEGAL)


def skel_token(t: str) -> str:
    """Consonant skeleton: survives transliteration ('saut prodkts' == 'south products')."""
    for a, b in SUBS:
        t = t.replace(a, b)
    t = re.sub(r"[aeiouy]", "", t)
    return re.sub(r"(.)\1+", r"\1", t)


def skel(normed_name: str) -> str:
    return skel_token(normed_name.replace(" ", ""))


# ----------------------------------------------------------------- addresses
_US = ("alabama al alaska ak arizona az arkansas ar california ca colorado co connecticut ct "
       "delaware de florida fl georgia ga hawaii hi idaho id illinois il indiana in iowa ia "
       "kansas ks kentucky ky louisiana la maine me maryland md massachusetts ma michigan mi "
       "minnesota mn mississippi ms missouri mo montana mt nebraska ne nevada nv ohio oh "
       "oklahoma ok oregon or pennsylvania pa tennessee tn texas tx utah ut vermont vt "
       "virginia va washington wa wisconsin wi wyoming wy")
_IN = ("maharashtra mh karnataka ka gujarat gj rajasthan rj kerala kl telangana tg bihar br "
       "odisha od orissa od punjab pb haryana hr delhi dl jharkhand jh chhattisgarh cg "
       "uttarakhand uk goa ga assam as")
_CITY = ("calcutta kolkata bombay mumbai madras chennai bangalore bengaluru gurgaon gurugram "
         "poona pune baroda vadodara trivandrum thiruvananthapuram mysore mysuru")
_STREET = ("street st str st saint st avenue ave av ave road rd court ct crt ct boulevard blvd "
           "bd blvd bld blvd drive dr lane ln place pl highway hwy parkway pkwy circle cir "
           "terrace ter square sq trail trl apartment apt suite ste floor fl building bldg "
           "sector sec sect sec chemin ch route rte allee all impasse imp north n south s "
           "east e west w first 1st second 2nd third 3rd fourth 4th fifth 5th sixth 6th "
           "seventh 7th eighth 8th ninth 9th tenth 10th")


def _pairs(s):
    t = s.split()
    return dict(zip(t[::2], t[1::2]))


ADDR_MAP = {**_pairs(_US), **_pairs(_IN), **_pairs(_CITY), **_pairs(_STREET)}
ADDR_DROP = frozenset("na n/a city near opp opposite behind beside c o h no po ps cedex".split())
PHRASES = [(f" {a} ", f" {b} ") for a, b in [
    ("new york", "ny"), ("new jersey", "nj"), ("new mexico", "nm"), ("new hampshire", "nh"),
    ("north carolina", "nc"), ("north dakota", "nd"), ("south carolina", "sc"),
    ("south dakota", "sd"), ("west virginia", "wv"), ("rhode island", "ri"),
    ("district of columbia", "dc"), ("west bengal", "wb"), ("uttar pradesh", "up"),
    ("madhya pradesh", "mp"), ("andhra pradesh", "ap"), ("himachal pradesh", "hp"),
    ("tamil nadu", "tn"), ("jammu and kashmir", "jk"), ("jammu kashmir", "jk"),
]]


def norm_addr(s: str) -> str:
    s = " " + re.sub(r"[^a-z0-9]+", " ", anyascii(s or "").lower()) + " "
    for a, b in PHRASES:
        if a in s:
            s = s.replace(a, b)
    out = []
    for t in s.split():
        t = ADDR_MAP.get(t, t)
        if t in ADDR_DROP:
            continue
        if t.isdigit():
            t = t.lstrip("0") or "0"
        out.append(t)
    return " ".join(out)
