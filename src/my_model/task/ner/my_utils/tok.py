def tok(line,tag='O'):
    SEP='\"\'0123456789!,.:;?@’([{)]}-^`_´¥¨~$”×<=&#|*&%+/─'
    li=[]
    s=[]

    line = line.strip()
    line=line.encode('utf-8').decode('utf-8-sig')
    for c in list(line):
        if c in ' ':
            if li:
                s.append(li)
                li = []
            continue
        if any([c==i for i in SEP]):
           if li:
                s.append(li)
                li = []
           s.append(['{}'.format(c)])
        else:
            li.append(c)
    if li:
        s.append(li)
    return [''.join(l) for l in s]
