#coding=utf8
import sys

def is_digit(c):
    if '0'<=c<='9':
        return True
    return False

def is_number(w,lang):
    # REF:https://en.wikipedia.org/wiki/Decimal_separator
    thousand_sep = ','
    decimal_sep = '.'
    if lang in ['pt','es','ru','mo','uk']:
        thousand_sep,decimal_sep = decimal_sep,thousand_sep
    # 1,234,567.89
    w2 = w.replace(thousand_sep,'').replace(decimal_sep,'')
    if w2 == '' or any(is_digit(c) == False for c in w2):
        return False
    if decimal_sep in w:
        if w.count(decimal_sep) > 1:
            return False
        idx = w.find(decimal_sep)
        int_part = w[:idx]
        if int_part == '':
            int_part = '0'
        dec_part = w[idx+1:]
        if any(is_digit(c) == False for c in dec_part):
            return False
    else:
        int_part = w
        dec_part = '0'
    if all(len(e) in [2,3] for e in int_part.split(thousand_sep)[1:]) and int_part[0] != thousand_sep:
        return True
    return False

def norm_space_for_num(s):
    o = ''
    ws = s.split()
    for i in range(len(ws)):
        w = ws[i]
        if all(is_digit(c) for c in w) and i+1 < len(ws) and len(ws[i+1]) >= 3 and all(is_digit(c) for c in ws[i+1][:3]):
            o += w
        else:
            o += w + ' '
    return o

def is_url_or_filename(w):
    key_identifier = ['http','www.','.com','.edu','.cn','.gov','.net','.org','.pdf','.doc','.txt','.ppt','.docx','.pptx']
    if any(e in w for e in key_identifier):
        return True
    return False

def proc_punc_lhs(w):
    punc_lhs = u'''$ £ € ¥ ؟ « 《 ' " ( [ { < 「 “ ‘'''.split()
    o = ''
    for i in range(len(w)):
        if w[i] in punc_lhs:
            o += w[i]+' '
        else:
            o += w[i:]
            break
    return o

def proc_punc_rhs(w):
    punc_rhs = u''', % ، ؟ » 》 ᠂ ᠃ ! ? ' " ; : ) ] } > 」 | । ” ’'''.split()
    o = ''
    for i in range(len(w))[::-1]:
        if w[i] in punc_rhs or w[i] == '.' and i-1>0 and w[i-1] in punc_rhs:
            o = ' '+w[i]+o
        else:
            o = w[:i+1] + o
            break
    return o

def proc_modal_verb(w):
    model_verb_abbrs = ["'s","'m","'re","'t","'d","'ll","'ve",u"’s",u"’m",u"’re",u"’t",u"’d",u"’ll",u"’ve"]
    for v in model_verb_abbrs:
        if w[-len(v):] == v:
            return w[:-len(v)] + ' ' + w[-len(v):]
    return w

def proc_unit(w,units,lang):
    for u in units:
        if w[-len(u):] == u and is_number(w[:-len(u)],lang):
            return w[:-len(u)] + ' ' + w[-len(u):]
    return w

def tok(s,nonbreaking_prefix,units,lang):
    s = s.decode('utf8','ignore')
    s = norm_space_for_num(s)
    pre_tok_ws = []
    for w in s.split():
        w = proc_punc_lhs(w)
        w = proc_punc_rhs(w)
        w = proc_modal_verb(w)
        pre_tok_ws += w.split()
    final_tok_ws = []
    for w in pre_tok_ws:
        if is_url_or_filename(w) or all(c=='.' for c in w) or w.lower() in nonbreaking_prefix:
            final_tok_ws.append(w)
        elif w[-1] == '.':
            w = proc_unit(w[:-1],units,lang)
            final_tok_ws += [w,'.']
        else:
            w = proc_unit(w,units,lang)
            final_tok_ws += [w]
    o = ' '.join(final_tok_ws).encode('utf8')
    return o

lang = 'en'
nonbreaking_prefix = open('nbp').read().decode('utf8').split()
units = open('units').read().decode('utf8').split()

for s in sys.stdin:
    print tok(s,nonbreaking_prefix,units,lang)
