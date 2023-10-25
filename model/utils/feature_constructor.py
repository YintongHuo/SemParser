import re

def StartsWithUppercaseFeature(token):
    if int(token[:1].istitle()):
        return [0,1]
    else:
        return [1,0]

def TokenLengthFeature(token):
    if len(token) >10:
        return [0,1]
    else:
        return [1,0]

def ContainsDigitsFeature(token):
    regexp_contains_digits = re.compile(r'[0-9]+')
    if regexp_contains_digits.search(token):
        return [0,1]
    else:
        return [1,0]

def ContainsPunctuationFeature(token):
    regexp_contains_punctuation = re.compile(r'[\.\,\:\;\(\)\[\]\?\!]+')
    if regexp_contains_punctuation.search(token):
        return [0,1]
    else:
        return [1,0]

def OnlyDigitsFeature(token):
    regexp_contains_only_digits = re.compile(r'^[0-9]+$')
    if regexp_contains_only_digits.search(token):
        return [0,1]
    else:
        return [1,0]

def OnlyPunctuationFeature(token):
    regexp_contains_only_punctuation = re.compile(r'^[\.\,\:\;\(\)\[\]\?\!]+$')
    if regexp_contains_only_punctuation.search(token):
        return [0,1]
    else:
        return [1,0]

def CamelFeature(token):
    camel = re.compile('^[a-z]+(?:[A-Z][a-z]+)*$')
    if camel.search(token):
        return [0,1]
    else:
        return [1,0]

def PascalFeature(token):
    pascal = re.compile('^[A-Z][a-z]+(?:[A-Z][a-z]+)*$')
    if pascal.search(token):
        return [0,1]
    else:
        return [1,0]

def extract_feature(token):
    swu = StartsWithUppercaseFeature(token)
    tlf = TokenLengthFeature(token)
    cdf = ContainsDigitsFeature(token)
    cpf = ContainsPunctuationFeature(token)
    odf = OnlyDigitsFeature(token)
    opf = OnlyPunctuationFeature(token)
    cf = CamelFeature(token)
    pf = PascalFeature(token)

    features = list()
    features.extend(swu)
    features.extend(tlf)
    features.extend(cdf)
    features.extend(cpf)
    features.extend(odf)
    features.extend(opf)
    features.extend(cf)
    features.extend(pf)

    return features

